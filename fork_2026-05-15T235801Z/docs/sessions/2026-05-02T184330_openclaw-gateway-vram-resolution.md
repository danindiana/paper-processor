# Session: OpenClaw Gateway VRAM Monopoly — Resolution

**Timestamp:** 2026-05-02T18:43:30-05:00  
**System:** worlock (Ubuntu/Debian, Linux 6.8.12)  
**Ollama:** v0.21.0 | **GPUs:** dual NVIDIA (RTX 3080 10G + RTX 3060 12G)

---

## Problem Statement

`ollama stop gpt-oss:20b` returned exit 0 but the model remained resident in VRAM (~16.6 GiB).
Every eviction attempt (including direct `keep_alive=0` API calls) was immediately undone by a
process continuously reloading the model.

---

## Investigation Summary

See full diagnostic log:
`docs/sessions/2026-05-02_gpt-oss-20b-eviction-investigation.md`

Key findings:

| Finding | Detail |
|---------|--------|
| Root cause | `openclaw-gateway` (PID 647427) held active ESTABLISHED TCP connection to Ollama port 11434 with a 7.7 MB kernel send buffer — live token generation in progress |
| Why `ollama stop` failed | Stop request queued behind active generation; model unloaded then immediately reloaded by openclaw's next queued request. `expires_at` reset to future timestamp, appearing as a no-op |
| Driver | OpenClaw cron job "Cron test event" (`~/.openclaw/cron/jobs.json`) firing every 60 seconds, invoking `main` agent against `gpt-oss:20b` |
| Service | `openclaw-gateway.service` — user systemd service, active since 2026-05-02 00:28, 16h+ uptime, 898 MB RAM, 16h CPU time |

---

## Actions Taken

### 1. Stopped openclaw-gateway service (2026-05-02 ~18:44)
```bash
systemctl --user stop openclaw-gateway
```
- Runner process PID 1218145 terminated
- `/api/ps` → `models: []`
- `nvidia-smi` → 183 MiB remaining (unrelated process)
- **16.5 GiB VRAM reclaimed**

### 2. Disabled the 60-second cron job (2026-05-02 ~18:44)
```diff
# ~/.openclaw/cron/jobs.json
-  "enabled": true,
+  "enabled": false,
```
Prevents automatic re-invocation of `gpt-oss:20b` if service is restarted.

### 3. Added directory motd (2026-05-02 ~18:45)
`~/.openclaw/.motd` — bash script showing live service/cron state.
Triggered by existing `chpwd` hook in `~/.zshrc` whenever user `cd`s into `~/.openclaw/`.

### 4. Added system motd entry (2026-05-02 ~18:45)
`/etc/update-motd.d/96-openclaw-status` — shows service/cron state + VRAM warning at login.

---

## Current State

| Resource | Before | After |
|----------|--------|-------|
| VRAM (gpt-oss:20b) | ~16.6 GiB | 0 GiB |
| Ollama runner process | running (PID 1218145, 148% CPU) | gone |
| openclaw-gateway service | active, 16h uptime | stopped |
| OpenClaw cron job | enabled, every 60 s | disabled |

---

## Re-enabling OpenClaw

```bash
# 1. Re-enable cron job (optional):
#    Edit ~/.openclaw/cron/jobs.json → "enabled": true

# 2. Start the service:
systemctl --user start openclaw-gateway

# 3. To also survive reboots:
systemctl --user enable openclaw-gateway   # was already enabled
```

**Warning:** if the cron job is re-enabled while service is running, `gpt-oss:20b` (17.9 GiB)
will reload into VRAM within 60 seconds. Either change the agent's model in
`~/.openclaw/agents/main/agent/models.json` to something smaller, or keep the cron disabled.

---

## Files Modified

| File | Change |
|------|--------|
| `~/.openclaw/cron/jobs.json` | `"enabled": false` on "Cron test event" job |
| `~/.openclaw/.motd` | created — directory cd notice |
| `/etc/update-motd.d/96-openclaw-status` | created — login motd entry |
