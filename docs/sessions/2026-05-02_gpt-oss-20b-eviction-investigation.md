# Investigation: gpt-oss:20b Not Evicting on `ollama stop`

**Date:** 2026-05-02T18:37:25-05:00  
**Resolved:** 2026-05-02T18:45:00-05:00  
**Ollama version:** 0.21.0  
**Model:** gpt-oss:20b (MXFP4, 20.9B params, 17.9 GiB VRAM)

---

## Symptom

Running `ollama stop gpt-oss:20b` does not remove the model from VRAM.
The command returns exit code 0 but the model remains resident.

---

## Environment

```
# /etc/systemd/system/ollama.service.d/override.conf
OLLAMA_HOST=127.0.0.1
CUDA_VISIBLE_DEVICES=0,1
OLLAMA_GPU_OVERHEAD=0
OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_KEEP_ALIVE=5m
```

Model runner commandline:
```
/usr/local/bin/ollama runner --ollama-engine \
  --model /usr/share/ollama/.ollama/models/blobs/sha256-e7b273f9... \
  --port 46457
```
Note: uses `--ollama-engine` (Ollama native inference, not llama.cpp).

---

## Investigation Sequence

### 1. Baseline snapshot

`/api/ps` confirmed model loaded with `expires_at: 2026-05-02T18:29:51-05:00`.  
Runner PID 1218145 at **148–149% CPU** (dual-core burn, not idle).  
VRAM via `nvidia-smi`: 10,200 MiB + 6,364 MiB = **~16.6 GiB** across both GPUs.

### 2. `ollama stop gpt-oss:20b`

```
$ ollama stop gpt-oss:20b; echo "exit: $?"
exit: 0
```

Model still in `/api/ps`. Runner still running. VRAM unchanged.  
**`expires_at` reset to a new future timestamp** (`18:36:27` — roughly 5–6 min ahead),
indicating the stop command sent a request that the model processed as a **timer refresh**
rather than an eviction.

### 3. Direct `keep_alive=0` API call

```
$ curl -X POST http://localhost:11434/api/generate \
    -H 'Content-Type: application/json' \
    -d '{"model":"gpt-oss:20b","keep_alive":0}'
{
  "model": "gpt-oss:20b",
  "done_reason": "unload",
  ...
}
```

Response says `done_reason: unload` — model reported as unloaded.  
Immediate `/api/ps` check: model **still present**, `expires_at` reset again to `18:36:50`
(the time our request completed). VRAM unchanged.

### 4. Active connection audit

```
$ ss -tp | grep 11434
ESTAB      0    7769200   127.0.0.1:11434   127.0.0.1:42096
ESTAB   3703439  0        127.0.0.1:42096   127.0.0.1:11434  ("openclaw-gatewa",pid=647427,fd=28)
CLOSE-WAIT  1    0        127.0.0.1:50816   127.0.0.1:11434  ("python",pid=826179,fd=3)
```

Two connections found:
- **`openclaw-gateway` (PID 647427)** — ESTABLISHED, kernel send buffer **7.7 MB and growing**
  (was 7.3 MB earlier). This is an active streaming generation.
- **`python paper_processor.py --override` (PID 826179)** — CLOSE_WAIT; client closed its
  end but Ollama has not finished.

`openclaw-gateway` is a user-space systemd service (parent: user systemd PID 1):
```
$ ps -p 647427 -o pid,ppid,comm,args --no-headers
647427    6621 openclaw-gatewa openclaw-gateway
```

---

## Root Cause

**`openclaw-gateway` is actively generating tokens from `gpt-oss:20b`**, holding an
established TCP connection with a growing backlog (7+ MB unsent data in the kernel buffer).

Ollama cannot evict a model while an in-flight generation is active on that model's runner.
The eviction request (`ollama stop` or `keep_alive=0`) is queued and processed when the
runner is free, but `openclaw-gateway` immediately submits the next request — so the model
reloads within milliseconds of any eviction.

### Why `ollama stop` appears to reset the timer

In v0.21.0, `ollama stop` sends a `keep_alive: 0` request to `/api/generate`. The runner
processes it, returns `done_reason: unload`, but `openclaw-gateway` already has the next
request queued. Ollama reloads the model to serve that request and sets a fresh `expires_at`
(current time + `OLLAMA_KEEP_ALIVE`). From the observer's perspective the stop "did nothing"
and the timer was refreshed.

### Secondary contributor: `OLLAMA_KEEP_ALIVE=5m`

The 5-minute global keep-alive is never reached because `openclaw-gateway` continuously
resets it. Even if `openclaw-gateway` paused, the model would linger for 5 minutes.

---

## Verified Non-Causes

| Hypothesis | Status |
|-----------|--------|
| `ollama stop` not a real command | **False** — present in v0.21.0, returns exit 0 |
| Ollama version too old | **False** — stop command exists and executes |
| `--ollama-engine` ignoring keep_alive | **Partially** — the unload path works; reload is the issue |
| MXFP4 memory-map leak | **Not confirmed** — model evicts correctly when no active client |

---

## How to Force Eviction Right Now

### Option A — kill the openclaw-gateway connection, then stop

```bash
# Identify and close the connection
kill -TERM 647427           # ask openclaw-gateway to exit gracefully
# or just drop the TCP connection:
ss -K dst 127.0.0.1 dport = 11434

# Then evict
curl -s -X POST http://localhost:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-oss:20b","keep_alive":0}'
```

### Option B — kill the runner directly

```bash
kill -TERM 1218145          # SIGTERM the runner; Ollama will restart it on next request
```

### Option C — restart Ollama (nuclear)

```bash
sudo systemctl restart ollama
```

---

## Resolution Applied (2026-05-02T18:45)

### Step 1 — Stopped openclaw-gateway service
```bash
systemctl --user stop openclaw-gateway
```
Result: runner process (PID 1218145) terminated immediately. `/api/ps` returned `models: []`.
`nvidia-smi` dropped from ~16.6 GiB to 183 MiB (unrelated process). VRAM fully reclaimed.

### Step 2 — Disabled the 60-second cron job
Edited `~/.openclaw/cron/jobs.json`: set `"enabled": false` on the "Cron test event" job.
This prevents openclaw from re-invoking `gpt-oss:20b` when the service is restarted.

---

## Recommended Fixes

### Short-term: evict when pipeline finishes

`paper_processor.py` already has `_ollama_evict()` (line ~119). Wire it as a post-pipeline
teardown so it fires after all sections complete, not just during `--override` startup.

### Medium-term: set explicit keep_alive in `_call_ollama()`

Add `"keep_alive": 60` (or shorter) to the generate payload in `_call_ollama()`. This limits
how long the model stays hot between sections without requiring explicit eviction calls.

### Long-term: investigate openclaw-gateway

Determine what workload `openclaw-gateway` is running against `gpt-oss:20b`. If it is
OpenWebUI or a similar frontend, the user may be unintentionally leaving a long chat
generation running. Consider lowering `OLLAMA_KEEP_ALIVE` to `1m` or `30s` in the systemd
override to reduce idle residency.
