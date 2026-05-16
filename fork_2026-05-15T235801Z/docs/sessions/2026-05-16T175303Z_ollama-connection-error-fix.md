# Session: Ollama ConnectionError Kills paper_processor Mid-Run

**Date:** 2026-05-16T175303Z  
**Fork:** `fork_2026-05-15T235801Z`  
**Branch:** main

---

## Symptom

`paper_processor.py` exited mid-run with:

```
📄  2501.14101v1.pdf
     pages=6  chunks=1
     model=deepseek-r1:8b
     code_model=qwen3-coder:30b
     strategy=single-pass (6 pages)
     📝  Summary …
❌  Cannot reach Ollama at http://localhost:11434
    Is `ollama serve` running?
```

Ollama was confirmed running immediately after (`systemctl status` showed active, `curl` responded).

---

## Investigation

### Service state

`systemctl status ollama` — active, running. `curl http://localhost:11434/api/version` — responded immediately. No IPv6 mismatch: Python resolves `localhost` to `127.0.0.1`, same as Ollama's bind address (`OLLAMA_HOST=127.0.0.1:11434`).

### Journal timeline (last hour)

```
12:35:16  Ollama stopped  (PID 1418430 — consumed 1h 12m CPU)
12:35:17  Ollama started  (PID 1960182)
12:35:29  HEAD /  +  GET /api/ps  (paper_processor polling)
12:35:37  Ollama stopped  (PID 1960182 — consumed only 1.041s CPU)
           ↑ second systemctl restart fired while first was still warm
12:37:17  Ollama started  (PID 4637)
12:38:32  runners loaded
12:39:15  llama runner ready (45.76 s load)
12:40:20–12:44:38  four generate requests (1–2 min each)
12:45:01–12:45:15  /api/ps polled every second (restart logic)
12:45:16  Ollama stopped + restarted again
```

During the **12:35:37–12:37:17 window** (~100 s), Ollama was unreachable. A `generate` call that landed in this window received a `ConnectionError`.

### Root cause

`_call_ollama` (paper_processor.py:448) had this handler:

```python
except requests.exceptions.ConnectionError:
    sys.exit(
        f"❌  Cannot reach Ollama at {OLLAMA_URL}\n"
        "    Is `ollama serve` running?"
    )
```

`sys.exit()` kills the entire process immediately with no retry. But the code itself calls `_ollama_restart_service()` on timeouts (line 456), which issues `sudo systemctl restart ollama` and creates a 20–100 s window where Ollama is genuinely unreachable. A `ConnectionError` during that window is a transient, self-inflicted condition — not a fatal misconfiguration — yet it was treated as unrecoverable.

The double-restart at 12:35:37 was a second `_ollama_restart_service()` call (likely from the restart-check path or a concurrent paper) re-killing the freshly-started service, extending the outage to ~100 s.

---

## Fix

**File:** `paper_processor.py:448`

Replaced the immediate `sys.exit()` with a wait-and-retry loop on attempt 1, falling back to `RuntimeError` (not `sys.exit`) on attempt 2:

```python
except requests.exceptions.ConnectionError as exc:
    if attempt == 1 and not _shutdown.is_set():
        print(f"     🔌  Ollama unreachable — waiting up to 60 s for it to come back …")
        deadline = time.time() + 60
        recovered = False
        while time.time() < deadline:
            time.sleep(3)
            try:
                requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).raise_for_status()
                recovered = True
                break
            except Exception:
                print(".", end="", flush=True)
        if recovered:
            print(" recovered!")
            continue
        print(" timed out")
    raise RuntimeError(
        f"❌  Cannot reach Ollama at {OLLAMA_URL} (model={model}): {exc}\n"
        "    Is `ollama serve` running?"
    ) from exc
```

**Behaviour change:**
- Attempt 1 `ConnectionError`: waits up to 60 s polling `/api/tags`, retries if Ollama recovers.
- Attempt 2 (or if 60 s elapses): raises `RuntimeError`, which the outer loop catches per-paper — logs the error and continues to the next paper rather than killing the whole run.
- `sys.exit()` is no longer called mid-run; all remaining `sys.exit` calls are at startup/validation time.

---

## Notes

- The 60 s wait window is intentionally longer than `_ollama_restart_service`'s 45 s poll deadline, covering the case where the first restart itself triggers a second restart.
- No changes to `_ollama_restart_service` — the double-restart race is a separate issue; the fix here makes the caller resilient to it.
- Ollama override warns that `flash_attention` and `kv_cache_type` options are invalid for the current version (WARN in journal) — these are harmless but could be cleaned up.
