# Session: Ollama Timeout Cascade Investigation
**Date:** 2026-05-02  
**Repo:** paper_processor  
**Branch:** main

---

## Observed Symptoms

Every paper in a batch run failed with:
```
Ollama error (model=gemma4:31b-it-q4_K_M):
HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read timeout=1200)
```

Even trivial papers (5 pages, `deepseek-r1:8b` model) timed out after apparently starting
their Summary section — despite `deepseek-r1:8b` normally completing in under a minute.

---

## Root Cause Analysis

### Primary cause: missing model triggers silent auto-download

`gemma4:31b-it-q4_K_M` was not present in the local Ollama library. When the first paper
issued a `/api/generate` request for it, Ollama silently began downloading the model
(~18-20 GB) before generating. With `stream: False`, the client blocks until the full
response arrives — meaning the 20-minute download looks identical to a slow generation.
The 1200-second client timeout fired before the download completed.

### Cascade mechanism

`OLLAMA_NUM_PARALLEL=1` serializes all requests to Ollama. After the Python client timed
out (closing its socket), the Ollama server continued downloading in the background.
Every subsequent paper's request queued behind the still-active download, waited its own
1200 seconds, and also timed out. This produced a cascade where *every* paper in the
batch failed, including ones using already-available small models.

### No inter-paper recovery

`_ollama_restart_service()` existed but was only called by the `--override` flag at
startup. After a timeout error in the main processing loop, the code simply moved to the
next paper — into a still-hung Ollama instance.

---

## Changes Made (`paper_processor.py`)

### 1. Pre-flight model check — `check_required_models()`

New function added near `health_check_ollama()`. Called from `main()` immediately after
the Ollama health check, before any PDF is opened.

Queries `/api/tags` and exits with explicit `ollama pull <model>` instructions if any
required model is absent. Prevents the entire batch from silently wasting hours on
auto-download timeouts.

Models checked: all unique models reachable via `TIER_BY_PAGES` + `CODE_MODEL`
(`deepseek-r1:8b`, `deepseek-r1:14b`, `gemma4:31b-it-q4_K_M`, `qwen3-coder:30b`).
With `--model OVERRIDE`, checks only that model + `CODE_MODEL`.

### 2. Retry with restart in `_call_ollama()`

Wrapped the HTTP request in a 2-attempt loop. On `requests.exceptions.Timeout` during
the first attempt, calls `_ollama_restart_service()` to kill the hung server state, then
retries once. The second timeout propagates as before.

Also improved HTTP error messages: non-2xx responses now extract Ollama's JSON `"error"`
field and surface it directly, instead of raising an opaque `HTTPError` status line.

### 3. Post-timeout Ollama restart in main processing loops

Both the serial loop and parallel `fut.result()` path now check if the exception message
contains `"timed out"`. If so, `_ollama_restart_service()` is called before the next
paper starts, clearing any stuck server state and preventing cascade failures.

---

## Verification Steps

```bash
# 1. Confirm which models are available
ollama list

# 2. Test pre-flight check with a missing model (should exit immediately with pull commands)
python paper_processor.py --backend ollama

# 3. Pull any missing models, then run single small paper
ollama pull gemma4:31b-it-q4_K_M
ollama pull qwen3-coder:30b
python paper_processor.py --paper "COMPUTING THE BUSY.pdf" --backend ollama

# 4. Full batch run with clean GPU state
python paper_processor.py --backend ollama --override
```

---

## Outstanding Notes

- The `gemma4:31b-it-q4_K_M` and `qwen3-coder:30b` models need to be pulled before the
  next batch run. The pre-flight check will now make this explicit rather than silently
  timing out.
- Consider adding `ollama pull` progress output to the pre-flight check in a future
  session if the user wants auto-pull with progress rather than a hard exit.
