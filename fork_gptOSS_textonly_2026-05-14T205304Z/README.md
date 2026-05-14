# paper_processor — fork_gptOSS_textonly_2026-05-14T205304Z

## What this fork is

A drop-in replacement for the parent `paper_processor.py` that swaps the
`xl_quality` model tier from `gemma4:31b-it-q4_K_M` to `gpt-oss:20b`.

## Why

`paper_processor.py` is a **text-only pipeline**:

- PDF pages are extracted with `fitz.get_text("text")` — all embedded images are discarded.
- Both backends (Ollama, OpenClaw) receive a plain `prompt: str` — no image fields, no base64, no vision parameters.

`gemma4:31b-it-q4_K_M` is a multimodal model. Those capabilities are never
exercised here, yet the model occupies ~18 GB VRAM on the dual-GPU pool.
`gpt-oss:20b` is text-only and fits in ~12 GB, recovering ~6 GB of headroom
for concurrent loads.

## What changed

| Location | Before | After |
|----------|--------|-------|
| `MODEL_TIERS["xl_quality"]` | `gemma4:31b-it-q4_K_M` | `gpt-oss:20b` |
| `--help` routing table | `gemma4:31b-q4_K_M (~18 GB)` | `gpt-oss:20b (~12 GB)` |

All routing thresholds, section prompts, backends, and checkpointing logic are
unchanged.

## Usage

Identical to the parent — activate the same venv and run:

```bash
source ../.venv/bin/activate
python paper_processor.py [OPTIONS] [INPUT_DIR]
```

Session documents for this fork live in `docs/sessions/`.
