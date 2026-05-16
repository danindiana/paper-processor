# Session: Graceful Shutdown Fix
**Date:** 2026-05-03  
**Repo:** paper_processor  
**Branch:** main

---

## Observed Symptoms

Pressing Ctrl+C during a batch run printed the "Shutdown requested" banner but the process
continued running — sometimes for tens of minutes — before actually stopping. Additionally,
on restart, sections that were interrupted mid-generation appeared as "complete" in
`metadata.json`, so they were skipped rather than re-run, leaving corrupted output on disk.

---

## Root Cause Analysis

### 1. Partial response written and marked complete

`_call_ollama()` used streaming (`stream: True`) and checked `_shutdown` inside the
token loop. But on shutdown it did:

```python
return "".join(full_response)   # ← returned partial tokens
```

The caller in `process()` received the truncated string, wrote it to disk with
`_write_md()`, and then appended the section to `completed` and checkpointed it as done.
On restart the section was skipped because it appeared complete. The output file contained
whatever fragment Ollama had emitted up to the signal — often a few sentences or nothing.

### 2. `map_reduce_chunks` used `break` instead of raising

`map_reduce_chunks` checked `_shutdown` between chunks and `break`-ed out of the loop,
returning whatever partial summaries existed (possibly an empty string). The calling code
in `process()` had no way to detect the early exit. It continued building `capped` from
the empty/partial context, then hit the first `_shutting_down()` inter-section check which
did stop — but only by luck. If a section had been in `completed` already, it would have
run a section against meaningless context.

### 3. `completed` and `_checkpoint` defined after `map_reduce_chunks`

If any exception escaped `map_reduce_chunks`, `_checkpoint` didn't exist yet, making it
impossible to save a checkpoint inside a `try/except` wrapped around the call.

### 4. `main()` logged clean shutdown as an error

Neither the serial nor parallel path in `main()` distinguished a user-requested shutdown
from a genuine processing failure. Both propagated through the bare `except Exception`
branch and appended the paper to the `errors` list, polluting the final summary.

---

## Changes Made (`paper_processor.py`)

### 1. `_ShutdownRequested` exception class

Added after `_install_signal_handlers()`:

```python
class _ShutdownRequested(Exception):
    """Raised when _shutdown fires during a blocking LLM call or map-reduce loop."""
```

All shutdown-triggered early exits now raise this instead of returning partial data or
silently breaking. This makes intent explicit and propagates cleanly through all callers.

### 2. `_call_ollama`: raise instead of return partial

```python
# before
return "".join(full_response)

# after
raise _ShutdownRequested()
```

Added `except (_ShutdownRequested, RuntimeError): raise` before the bare
`except Exception` catch-all so `_ShutdownRequested` isn't wrapped into a `RuntimeError`.

### 3. `map_reduce_chunks`: raise instead of break

```python
# before
if _shutdown.is_set():
    break

# after
if _shutdown.is_set():
    raise _ShutdownRequested()
```

### 4. `process()`: move `completed`/`_checkpoint` before `map_reduce_chunks`

Moved `completed`, `_checkpoint()`, and `_shutting_down()` definitions to immediately
after `paper_hash` is computed — before the `map_reduce_chunks` call — so a checkpoint
can always be saved in the event of an early exit from the map-reduce phase.

Wrapped `map_reduce_chunks` in an explicit try/except:

```python
try:
    context = map_reduce_chunks(chunks, self.backend, model)
except _ShutdownRequested:
    _checkpoint()
    print(f"     ⚡  Stopped during map-reduce — {len(completed)} section(s) saved")
    return
```

### 5. Per-section `_ShutdownRequested` guard

Each section's `backend.call()` is now wrapped so a mid-generation interrupt checkpoints
and returns without marking the section complete:

```python
try:
    out = self.backend.call(...)
except _ShutdownRequested:
    _checkpoint()
    print(f"     ⚡  Interrupted mid-section — {len(completed)} section(s) saved")
    return
# write to disk and mark complete only if we get here
self._write_md(...)
completed.append(section)
_checkpoint()
```

### 6. `main()`: clean shutdown not logged as error

Both serial and parallel paths now catch `_ShutdownRequested` before `except Exception`:

```python
# serial
except _ShutdownRequested:
    break

# parallel fut.result()
except _ShutdownRequested:
    pass
```

---

## Shutdown Behaviour After Fix

| When Ctrl+C fires | What happens |
|---|---|
| Between sections | `_shutting_down()` catches it; checkpoints; returns cleanly |
| During `map_reduce_chunks` (between chunk calls) | `_ShutdownRequested` raised; checkpoint saved; paper skipped |
| During `map_reduce_chunks` (mid-chunk stream) | `_ShutdownRequested` raised from `_call_ollama`; same path as above |
| During a section's LLM call (mid-stream) | `_ShutdownRequested` raised; section NOT written or marked complete; checkpoint of prior sections saved |
| Second Ctrl+C | `os._exit(1)` — immediate forced exit (pre-existing behaviour) |

On restart, only fully completed sections are in `metadata.json`; any interrupted section
is re-run from scratch with correct context.

---

## Verification

```bash
# Start a batch run then hit Ctrl+C mid-generation
python paper_processor.py --backend ollama

# Inspect checkpoint — interrupted section should NOT appear in sections_completed
cat _processed/<slug>/metadata.json | python3 -m json.tool

# Re-run — should resume from the interrupted section, not re-do completed ones
python paper_processor.py --paper "<filename>.pdf" --backend ollama
```
