# Future Directions & TODO

**Last updated:** 2026-05-10
**Branch:** `fork_2026-05-09T184929Z` — Dual-GPU Residency Fork

This document captures improvements identified through empirical testing and code review.
Items are grouped by theme and ranked within each group by impact-to-effort ratio.

---

## 1. Throughput & GPU Utilisation

### 1a. Parallel sections after map-reduce *(High impact / Medium effort)*
**Status:** TODO

After `map_reduce_chunks()` completes, all five sections (`summary`, `logic`, `cpp`,
`diagrams`, `extras`) independently consume the same `capped` context string.
Currently they run in series; `logic`, `cpp`, `diagrams`, and `extras` have zero
dependencies on each other and could fan out concurrently after `summary` completes.

Expected win: ~60% reduction in per-paper wall time on GPU 1 (RTX 5080,
`OLLAMA_NUM_PARALLEL=2`). On GPU 0 the Ollama service serialises requests anyway
(`OLLAMA_NUM_PARALLEL=1`), so the gain is smaller there unless sections are routed
across GPUs.

Implementation sketch:
```python
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = {
        "logic":   ex.submit(run_section, "logic",   ...),
        "cpp":     ex.submit(run_section, "cpp",     ...),
        "diagrams":ex.submit(run_section, "diagrams",...),
        "extras":  ex.submit(run_section, "extras",  ...),
    }
    for name, f in futs.items():
        f.result()  # propagate exceptions
```

Complications: checkpoint / `completed` list must be thread-safe; section writers
must use their own file handles; `_shutdown` signal must cancel the inner pool.

---

### 1b. Default `--workers` to 2 *(High impact / Trivial)*
**Status:** DONE (2026-05-10)

GPU 1 sits completely idle when `--workers 1` (the old default). Changed default
to `2` so both GPUs are utilised from the first invocation without requiring a flag.

---

### 1c. Route long papers to GPU 1 *(Medium-High impact / Low effort)*
**Status:** DONE (2026-05-10)

GPU 0 (RTX 3080) is hard-capped at 8 192 context tokens; GPU 1 (RTX 5080) supports
32 768. Papers >20 pages forced onto GPU 0 receive a heavily compressed map-reduce
and produce lower-quality output. The worker dispatcher now prefers GPU 1 for papers
with >20 pages when both GPUs are free.

---

## 2. Model Selection & Quality

### 2a. Diagram section uses code model *(Medium impact / Trivial)*
**Status:** DONE (2026-05-10)

Previously `DIAGRAM_PROMPT` was called with the reasoning model (`deepseek-r1:8b`).
Generating valid Graphviz DOT is a structured-output / code task; switched to the
code model (`qwen2.5-coder:7b` on GPU 0, `ministral-3:8b` on GPU 1). Produces
fewer delimiter and syntax errors.

---

### 2b. Tiered model selection for section complexity *(Medium impact / Low effort)*
**Status:** TODO

Not all sections need the full reasoning model. `extras` (critical analysis) benefits
from deep reasoning; `logic` (symbolic notation) is largely mechanical. A three-tier
dispatch table would cut token spend and latency on lighter sections:

| Section   | Suggested tier | Rationale |
|-----------|----------------|-----------|
| summary   | reason         | Requires synthesis across chunks |
| logic     | fast           | Mechanical symbolic transcription |
| cpp       | code           | Already uses code model |
| diagrams  | code           | Already switched (2a above) |
| extras    | reason         | Requires evaluation and critique |

---

### 2c. Adaptive chunk size based on page density *(Low impact / Medium effort)*
**Status:** TODO

Current sliding window (`window=12, overlap=2`) is fixed regardless of page density
(a dense methods section vs a sparse appendix). A density-aware chunker that targets
~3 000 tokens per chunk rather than 12 pages would produce more uniform summaries and
reduce wasted context.

---

## 3. VRAM Residency & Cold-Start

### 3a. Pre-warm models at startup *(Medium impact / Low effort)*
**Status:** DONE (2026-05-10)

Added `_prewarm_models()` at startup: sends a 1-token ping inference to each
model on each GPU before the first paper is processed. Eliminates the 10–30 s
cold-load penalty on the first paper.

---

### 3b. Keep-alive heartbeat thread *(Low-Medium impact / Low effort)*
**Status:** TODO

`OLLAMA_KEEP_ALIVE=5m` evicts models after 5 minutes of inactivity. During a large
batch run, slow papers (>5 min each) cause model eviction between papers. A
background daemon thread that pings each GPU every 4 minutes keeps both models
resident across the entire run without touching systemd config.

Implementation: `threading.Thread(target=_heartbeat, daemon=True).start()` in
`main()`, where `_heartbeat` loops over `BACKEND_URLS`, POSTs a 1-token request,
and sleeps 240 s.

---

### 3c. Explicit model eviction between incompatible runs *(Low impact / Trivial)*
**Status:** TODO

When switching between very different model configurations (e.g., testing a new
model), leftover resident models consume VRAM silently. A `--clean` flag that calls
`keep_alive=0` on all known models before starting would provide a clean slate
without requiring `--override` (which restarts the service).

---

## 4. DOT Diagram Robustness *(resolved 2026-05-10)*

All known diagram rendering failures have been fixed as of session
`2026-05-10T132822Z`. Summary of fixes applied:

| Fix | Description |
|-----|-------------|
| Hex-only prompt | Removed ambiguous colour name labels; LLM now uses only quoted hex values |
| Stderr logging | `render_dot()` now prints graphviz error messages on failure |
| `sanitize_dot()` | Regex replaces any surviving bare colour names with hex equivalents |
| Tolerant `_DELIM_RE` | Closing delimiter pattern changed from exact `===DIAGRAM_END===` to `===DIAGRAM_END[^\n]*` to tolerate LLM typos |
| Brace truncation | `sanitize_dot()` truncates content after the last `}` to strip leaked delimiter text |

Remaining diagram work:
- **Batch reprocess:** Run `--reprocess diagrams --workers 2` over all 101 papers
  that currently have `_raw_llm_output.txt` instead of rendered SVGs.
- **SVG viewer:** A lightweight `index.html` in each paper's `diagrams/` directory
  showing all 6 SVGs would make review much faster.

---

## 5. Observability & Operations

### 5a. Per-paper timing log *(Low impact / Low effort)*
**Status:** TODO

No timing data is currently emitted. Recording wall-clock time per section and
appending it to `metadata.json` would make it easy to identify slow papers and
benchmark the effect of the parallel-sections improvement (item 1a).

### 5b. Batch status dashboard *(Low impact / Medium effort)*
**Status:** TODO

`--list` shows completion status but no timing, error counts, or SVG render rates.
A richer summary (total papers, sections complete, SVGs missing, average time/paper)
would make it easier to monitor a long batch run.

### 5c. Retry failed sections automatically *(Low impact / Low effort)*
**Status:** TODO

Currently a failed section (Ollama timeout, empty response) is silently skipped.
Adding a single-retry with a brief back-off before writing the failure to disk would
recover from transient Ollama contention without manual `--reprocess`.

---

## Implementation Priority

| Priority | Item | Effort |
|----------|------|--------|
| ✅ Done  | 1b — default workers=2 | Trivial |
| ✅ Done  | 1c — route long papers → GPU 1 | Low |
| ✅ Done  | 2a — diagrams use code model | Trivial |
| ✅ Done  | 3a — pre-warm models at startup | Low |
| **Next** | 3b — keep-alive heartbeat | Low |
| **Next** | 5a — per-paper timing in metadata | Low |
| **Future** | 1a — parallel sections | Medium |
| **Future** | 2b — tiered model selection | Low |
| **Future** | 4 — batch reprocess diagrams | Ops task |
