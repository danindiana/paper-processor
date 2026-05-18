# Session: Remove Invalid Per-Request Ollama Options

**Date:** 2026-05-18T031953Z  
**Fork:** `fork_2026-05-15T235801Z`  
**Branch:** main  
**Commit:** 49b1c20

---

## Symptom

Ollama journal flooded with repeated warnings on every inference call:

```
level=WARN source=types.go:991 msg="invalid option provided" option=flash_attention
level=WARN source=types.go:991 msg="invalid option provided" option=kv_cache_type
```

---

## Investigation

### Source in code

`_call_ollama()` (paper_processor.py:408–411) was appending these fields to the
`options` dict sent to `/api/generate`:

```python
if self.flash_attn:
    options["flash_attention"] = True
if self.kv_cache_type:
    options["kv_cache_type"] = self.kv_cache_type
```

### Root cause

Ollama 0.22.1 validates the options dict in `types.go:991` and rejects both keys
as unknown. They are **load-time runner settings**, not per-request inference
parameters. `pgrep` confirmed the runner subprocess is launched with
`--ollama-engine`; that runner has no per-request hook for these values.

The correct mechanism is server-level environment variables consumed at model
load time:

| Rejected per-request option | Correct env var            |
|-----------------------------|----------------------------|
| `flash_attention`           | `OLLAMA_FLASH_ATTENTION=1` |
| `kv_cache_type`             | `OLLAMA_KV_CACHE_TYPE=q8_0`|

Ollama's journal confirmed the env-var path works — after setting them, every
subsequent load request showed `FlashAttention:Enabled KvCacheType:q8_0` in the
runner's load log line.

---

## Fix

**`paper_processor.py`** — removed both conditional inserts from the options
dict and replaced with a comment pointing to the env-var approach.

**`/etc/systemd/system/ollama.service.d/override.conf`** — added:

```ini
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
```

Ollama was restarted with the new environment; server config log confirmed
`OLLAMA_FLASH_ATTENTION:true` and `OLLAMA_KV_CACHE_TYPE:q8_0` active.

---

## Side effects

The CLI flags `--no-flash` and `--kv-cache` remain in the argument parser but
now have no effect on Ollama inference. Flash attention and KV cache type are
controlled entirely by the systemd env vars. The flags can be removed in a
future cleanup pass.
