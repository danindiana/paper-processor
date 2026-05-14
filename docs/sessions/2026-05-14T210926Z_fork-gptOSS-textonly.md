# Session: fork_gptOSS_textonly — creation, test, rename

**Date:** 2026-05-14  
**Operator:** jeb  
**Host:** worlock (192.168.1.151)

## Summary

Created `fork_gptOSS_textonly_2026-05-14T205304Z` — a drop-in variant of
`paper_processor.py` that replaces `gemma4:31b-it-q4_K_M` with `gpt-oss:20b`
as the `xl_quality` model tier.

## Motivation

Code review confirmed the pipeline is entirely text-only:

- `fitz.get_text("text")` discards all embedded PDF images.
- `Backend.call(prompt: str, ...)` passes only a plain text string to Ollama.
- No multimodal fields (`image`, `base64`, vision params) exist in either
  the Ollama or OpenClaw request payloads.

`gemma4:31b-it-q4_K_M` is a multimodal model holding ~18 GB VRAM for
capabilities the pipeline never uses. `gpt-oss:20b` is text-only at ~12 GB,
recovering ~6 GB of headroom on the dual-GPU pool (RTX 5080 + RTX 3080).

## Changes made

| File | Change |
|------|--------|
| `fork/paper_processor.py` | `MODEL_TIERS["xl_quality"]` → `gpt-oss:20b` |
| `fork/paper_processor.py` | `--help` routing table updated to match |
| `fork/README.md` | Describes fork purpose and what changed |
| `fork/docs/sessions/` | This operational notebook |
| root `README.md` | Added **Forks** table |

## Test run

Processed `019-3-034396-CT2578.pdf` (22 pages → routes to `xl_quality`):

```
model=gpt-oss:20b  code_model=qwen3-coder:30b  strategy=sliding-window (2 chunks)
✓ Summary  ✓ Symbolic logic  ✓ C++ examples  ✓ 6 diagrams  ✓ Extras
Output → _processed/019-3-034396-ct2578/
```

All five sections and six SVG diagrams completed without error.

## Naming

Fork was initially created as `fork_2026-05-14T205304Z`, then renamed to
`fork_gptOSS_textonly_2026-05-14T205304Z` for human readability.
`git mv` was used so history is preserved.

## venv

The fork shares the parent `.venv` (Python 3.13.7). From inside the fork
directory, `v` (alias for `source /home/jeb/bin/v.sh`) resolves `../.venv`
automatically — no separate venv needed.

## Commits

```
f4a266a  feat: fork_2026-05-14T205304Z — replace gemma4 with gpt-oss:20b
1646ba9  refactor: rename fork to fork_gptOSS_textonly_2026-05-14T205304Z
```
