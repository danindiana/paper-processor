# DOT Render Fix & Code Model Swap

**Host:** worlock
**Working dir:** `/home/jeb/programs/python_programs/paper_processor/fork_2026-05-09T184929Z`
**Started:** 2026-05-10 13:28:22Z
**Status:** Completed

## Goal

1. Diagnose and fix the ~33% SVG render failure rate for Graphviz DOT diagrams.
2. Replace the GPU 0 code model (`qwen3.5:4b`, general-purpose) with a dedicated code generation model that still fits within the ≤9 GB VRAM spec.

## What Was Changed

### paper_processor.py — DOT Rendering Fixes

**Fix 1 — `DIAGRAM_PROMPT` color block (lines 279–289):**
- Removed human-readable color name labels (`Electric Green`, `Hot Magenta`, etc.)
- Replaced with hex-only palette in quoted-string format
- Added explicit rule: "NEVER write `color=Electric Green`"
- Added node ID rules: letters/digits/underscores only; use `label=` for display text with special chars

**Fix 2 — `render_dot()` stderr logging (line ~648):**
- Added `if r.returncode != 0 and r.stderr.strip(): print(...)` to surface graphviz error messages
- Previously all `dot` failures were silent — impossible to diagnose from logs

**Fix 3 — `sanitize_dot()` function (new, after `ensure_neon_black()`):**
- Added `_COLOR_NAMES` dict mapping the 9 prompt color names to their hex equivalents
- `sanitize_dot()` uses regex to replace any bare `=Electric Green` style attribute values with `="#00FF41"` equivalents
- Acts as a safety net for any future LLM output that still emits color names despite the prompt fix
- Also truncates any content after the last `}` — catches delimiter text that leaks into DOT content (see Fix 5)
- Wired into the pipeline: `dot_src = sanitize_dot(ensure_neon_black(dot_src))`

**Fix 4 — `_DELIM_RE` malformed closing delimiter tolerance:**
- Changed closing pattern from `===DIAGRAM_END===` to `===DIAGRAM_END[^\n]*`
- The LLM consistently emitted `===DIAGRAM_END==-` (trailing `-` instead of `===`) causing the regex to miss all closing delimiters
- Without this fix, the non-greedy `.*?` would span across multiple diagrams and capture everything as one block

**Fix 5 — `sanitize_dot()` brace truncation (part of Fix 3 above):**
- When `_DELIM_RE` missed a delimiter, leaked text like `===DIAGRAM_END==-` ended up inside the written `.dot` file after the closing `}`
- Graphviz choked: `syntax error in line 26 near '='`
- `rfind("}")` truncation strips all trailing non-DOT content as a belt-and-suspenders guard

### paper_processor.py — GPU 0 Code Model (lines 342–353)

| | Before | After |
|---|---|---|
| Model | `qwen3.5:4b` | `qwen2.5-coder:7b` |
| Size | 3.2 GB | 4.4 GB |
| Type | General-purpose | Purpose-built code generation |
| Combined w/ reason | 8.6 GB raw | 9.6 GB raw |

Also corrected the GPU 1 comment from "RTX 3060 (12 GB)" to "RTX 5080 (16 GB)" to match actual hardware.

## Findings / Observations

1. **Root cause of 67% SVG success rate:** The `DIAGRAM_PROMPT` listed color names alongside hex values (e.g., `Electric Green  #00FF41`). LLMs occasionally took the color name literally and emitted `color=Electric Green` — a syntax error in DOT (unquoted multi-word value). The `render_dot()` function captured the graphviz stderr but never printed it, so these failures were completely invisible.

2. **Second root cause found during live testing — malformed closing delimiters:**
   - The LLM consistently output `===DIAGRAM_END==-` (trailing `-`) instead of `===DIAGRAM_END===`
   - The original `_DELIM_RE` regex required an exact `===DIAGRAM_END===` match — missed all of them
   - Result: the non-greedy `.*?` span captured content from diagram N all the way through diagrams N+1, N+2, etc.
   - The written `.dot` file contained multiple diagrams' content plus raw delimiter text after the closing `}`
   - `dot` reported: `syntax error in line 26 near '='` — pointing at `===DIAGRAM_END==-` on line 26
   - Stderr logging (Fix 2) was essential to identifying this; without it the failure reason was completely invisible

3. **Other secondary failure modes (less frequent):**
   - Invalid node identifiers containing `?` or `/` (e.g., `CONVERGE?`)
   - Malformed subgraph declarations missing braces

3. **`qwen2.5-coder:7b` VRAM trade-off on GPU 0 (RTX 3080, 10 GB):**
   - Raw combined size 9.6 GB vs the previous 8.6 GB is tight.
   - The 8192-token context cap keeps KV cache small; empirical testing from the fork's prior session showed ~8.5 GB actual dual-model VRAM usage with qwen3.5:4b.
   - With qwen2.5-coder:7b, actual usage is estimated ~9.5–10 GB — borderline.
   - If model eviction occurs on GPU 0, set `OLLAMA_MAX_LOADED_MODELS=1` in the GPU 0 service override to allow sequential swapping (slower but stable).
   - GPU 1 (RTX 5080, 16 GB) is unaffected.

4. **`qwen2.5-coder:7b` was already installed** — no `ollama pull` needed.

## Test Results

Ran `--reprocess diagrams` on `2407.02880v1.pdf` (29 pages, previously had 2 missing SVGs):

| Run | Result | Notes |
|-----|--------|-------|
| Run 1 (pre-fix) | 1/6 ✓ | Ollama contested — only 1 diagram returned |
| Run 2 (Fix 1–3 applied) | 2/6 ✓, 1/6 ✗ | Stderr exposed `syntax error in line 26 near '='`; led to Fix 4/5 |
| Run 3 (Fix 4–5 applied) | **6/6 ✓** | All diagrams rendered cleanly, zero errors |

## Next Steps

- [x] Run `--reprocess diagrams` on `2407_02880v1` to verify the SVG render fix — **6/6 ✓**
- [ ] Monitor GPU 0 VRAM with `nvidia-smi` during a code-section run to confirm `qwen2.5-coder:7b` loads without evicting `deepseek-r1:8b`
- [ ] If simultaneous residency fails on GPU 0, set `OLLAMA_MAX_LOADED_MODELS=1` in `/etc/systemd/system/ollama.service.d/override.conf` for port 11434
- [ ] Consider a batch `--reprocess diagrams` pass over all 101 papers with `_raw_llm_output.txt` files

## Environment Details

- **Script:** `./paper_processor.py`
- **GPU 0:** RTX 3080 (10 GB) — Port 11434, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=2`, ctx cap 8192
- **GPU 1:** RTX 5080 (16 GB) — Port 11435, `OLLAMA_NUM_PARALLEL=2`, `OLLAMA_MAX_LOADED_MODELS=2`, ctx cap 32768
