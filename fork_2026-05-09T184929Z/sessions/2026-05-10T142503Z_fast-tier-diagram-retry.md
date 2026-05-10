# Session: Fast-Tier Diagram Retry Fix

**Date:** 2026-05-10T14:25:03Z
**Branch:** `fork_2026-05-09T184929Z` — Dual-GPU Residency Fork

---

## Problem

Short papers (≤10 pages) were producing `diagrams/_raw_llm_output.txt` instead of rendered SVGs. Inspecting two known failures:

| Paper | Pages | Diagram time | Raw output |
|-------|-------|-------------|------------|
| `s41598-020-71287-1` | 10 | 13.4 s | Prose summary (neuroscience/BCI) |
| `728_25_18_1` | 8 | ~12 s | Prose summary (crustacean biology) |

The 13–14 s wall time is far too fast for generating 6 full DOT diagrams. The model returned a brief prose summary and stopped — it ignored `DIAGRAM_PROMPT` entirely.

## Root Cause

The diagram section calls `code_model` (`qwen2.5-coder:7b` on GPU 0, `ministral-3:8b` on GPU 1). These code-focused models handle structured output well for AI/ML papers (which dominate the training corpus), but on non-ML topics (neuroscience, biology) they default to their instruction-following persona and produce prose instead of DOT code.

`parse_diagrams()` finds no `===DIAGRAM_START:` delimiters, returns `[]`, and the raw response is saved to disk. The failure was silent — no retry, no escalation.

## Fix

Added a single retry inside `_run_diagrams()` when `parse_diagrams(raw)` returns empty:

1. Print `↺ No DOT blocks found; retrying with reason model …`
2. Prepend a forceful one-liner to `DIAGRAM_PROMPT`:  
   `"OUTPUT ONLY the 6 delimited Graphviz DOT blocks below. Do NOT write prose, summaries, or explanations. Start your first line with ===DIAGRAM_START:"`
3. Call `self.backend.call(...)` with `get_gpu_model("xl_quality")` (`deepseek-r1:8b`) instead of `code_model`
4. Re-run `parse_diagrams()` on the retry response
5. Only write `_raw_llm_output.txt` if the retry also returns empty

**Why `reason_model`?** `deepseek-r1:8b` is trained on RLHF with strong instruction-following for structured tasks. It produced all 6 DOT diagrams reliably in earlier sessions (including on the non-ML test paper `2407.02880`).

## Code Change

**File:** `paper_processor.py`  
**Function:** `_run_diagrams()` closure in `process()` (~line 907)

```python
diagrams = parse_diagrams(raw)
if not diagrams and not _shutdown.is_set():
    print("     ↺   No DOT blocks found; retrying with reason model …")
    _retry_prompt = (
        "OUTPUT ONLY the 6 delimited Graphviz DOT blocks below. "
        "Do NOT write prose, summaries, or explanations. "
        "Start your first line with ===DIAGRAM_START:\n\n"
        + DIAGRAM_PROMPT
    )
    raw = self.backend.call(
        self._tag_prompt(_retry_prompt, capped[:60_000]),
        get_gpu_model("xl_quality"),   # deepseek-r1:8b on both GPUs
        ctx_tokens=32768,
    )
    diagrams = parse_diagrams(raw)
if not diagrams:
    # write _raw_llm_output.txt as before
```

## Behaviour After Fix

| Scenario | Before | After |
|----------|--------|-------|
| Code model returns DOT | ✓ SVGs rendered | ✓ unchanged |
| Code model returns prose | ✗ `_raw_llm_output.txt` | ↺ retry → ✓ SVGs (reason model) |
| Both models return prose | ✗ `_raw_llm_output.txt` | ✗ `_raw_llm_output.txt` (no worse) |
| `_shutdown` set mid-run | ✓ exits cleanly | ✓ retry skipped if shutdown set |

## Next Steps

- Run `--reprocess diagrams --workers 2` over papers with existing `_raw_llm_output.txt` to regenerate SVGs
- If non-ML papers still fail after retry, consider a domain-specific fallback prompt or marking them as "diagram-unsupported"
