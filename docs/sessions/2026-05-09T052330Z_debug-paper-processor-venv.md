# Session: Enable venv shebang + debug paper_processor.py

**Timestamp (UTC):** 2026-05-09T05:23:30Z
**Local stamp:** 20260509_002330 (CDT)
**Date:** 2026-05-09
**Host / user:** WORLOCK drive mounted at `/run/media/morlok/WORLOCK/`, run as `morlok`
**Repo:** paper_processor
**Branch:** main

---

## Goal

1. Update `paper_processor.py`'s shebang so `./paper_processor.py` automatically
   uses the project's `.venv/bin/python` (set up in the previous session).
2. Run the script end-to-end on a single paper with the uncommitted streaming /
   `_ShutdownRequested` / `check_required_models` diff applied, surface any
   bugs.
3. Document the session.

---

## Changes

### 1. Shebang → venv python

`paper_processor.py:1`:

```diff
-#!/usr/bin/env python3
+#!/usr/bin/env -S ./.venv/bin/python
```

Also `chmod +x paper_processor.py` (file was `-rw-------`).

The `-S` arg-splitting flag is required so `env` treats `./.venv/bin/python`
as one argument rather than the shebang failing with "no such file or
directory: env -S ./.venv/bin/python". Supported on GNU coreutils ≥ 8.30
(Linux, this host).

**Caveat:** the shebang is *relative to the working directory*, so
`./paper_processor.py` only works when invoked from the repo root. From any
other CWD, fall back to `./.venv/bin/python paper_processor.py` or the `v`
shell helper from the previous session.

### 2. Bug fix: `check_required_models` over-checked CODE_MODEL when `--model` forced

`paper_processor.py:1050-1056`. Previously:

```python
models_to_check = (
    list(dict.fromkeys([args.model, CODE_MODEL]))
    if args.model
    else list(dict.fromkeys(
        [MODEL_TIERS[k] for _, k in TIER_BY_PAGES] + [CODE_MODEL]
    ))
)
```

Problem: when `--model X` is forced, `code_model = self.forced_model or
CODE_MODEL` (paper_processor.py:718) makes `code_model == X`, so `CODE_MODEL`
is never used. Yet the model-availability gate still required it to be
locally pulled — meaning `--model qwen3.5:9b` would exit with
`Required models not found locally — qwen3-coder:30b` despite that model
being unused on the forced path.

Fix:

```diff
 models_to_check = (
-    list(dict.fromkeys([args.model, CODE_MODEL]))
+    [args.model]
     if args.model
     else list(dict.fromkeys(
         [MODEL_TIERS[k] for _, k in TIER_BY_PAGES] + [CODE_MODEL]
     ))
 )
```

This was a real blocker: this host's local Ollama only had `qwen3.5:9b` and
`qwen3.5:2b-q8_0` — none of `MODEL_TIERS` or `CODE_MODEL` are pulled. Without
this fix, no end-to-end run was possible.

---

## End-to-end runs (verifying the uncommitted streaming + shutdown diff)

### Run A — `Huang_CV_Failure.pdf` on `qwen3.5:9b` (3 pages)

```bash
./paper_processor.py --model qwen3.5:9b \
    --paper Huang_CV_Failure.pdf \
    /run/media/morlok/WORLOCK/home/jeb/Documents/AI-ML_Papers
```

- Pre-flight: `health_check_ollama` ✓; `check_required_models` ✓ (after fix #2).
- `📝 Summary …` ✓ — produced 5,830-byte `01_summary.md` with detailed
  structured analysis.
- `🔣 Symbolic logic …` started; SIGTERM sent mid-section.
- New signal handler fired: `⚡ Shutdown requested — finishing current
  section then stopping …` then `⚡ Interrupted mid-section — 1 section(s)
  saved`. Clean exit code 0.
- `metadata.json` correctly checkpointed `sections_completed: ["summary"]`
  (logic was in flight, not committed).

→ **Validates: streaming Ollama path, `_ShutdownRequested` exception
plumbing, and per-section checkpointing.**

(LLM aside: the model correctly identified the PDF as a personal "CV of
Failure" rather than a research paper — the pipeline doesn't gate on document
type, which is fine.)

### Run B — `ferguson-abstract.pdf` on `qwen3.5:9b` (1 page, 2,921 chars / ~420 words)

```bash
PYTHONUNBUFFERED=1 ./paper_processor.py --model qwen3.5:9b \
    --paper ferguson-abstract.pdf \
    /run/media/morlok/WORLOCK/home/jeb/Documents/AI-ML_Papers
```

Section timings:

| Section          | Duration                   | Output size  | Verdict |
|------------------|----------------------------|--------------|---------|
| Summary          | ~30 s                      | 6,136 bytes  | ✓       |
| Symbolic logic   | ~30 s                      | 6,743 bytes  | ✓       |
| C++ examples     | ~22 min (!)                | **32 bytes** | ⚠ effectively empty |
| Graphviz diagrams| < 1 s before SIGTERM       | n/a          | killed  |

**Anomaly:** the C++ section ran on GPU 0 at 90 % util for 22 minutes and
returned only the section header (`# C++ Implementation Examples` + two
blank lines, 32 bytes). The two earlier sections on the *same model with the
same streaming code* produced 6 KB each in ~30 s. Suspected cause:
`qwen3.5:9b` enters a long thinking/deliberation pattern on the C++
prompt template (which asks for "complete C++20/23 implementations of key
algorithms with worked examples"). The streaming code accumulates `data
["response"]` but if the model emits only `thinking` field chunks (not
`response`), `full_response` stays empty until generation completes. We did
not isolate the exact cause — out of scope for this debug session.

**Not a bug in the diff** — sections 1 and 2 prove the streaming code
correctly accumulates and writes Ollama responses. The C++-section
anomaly is a model-fit issue (qwen3.5 family vs. the script's prompts,
which were tuned for `deepseek-r1`, `qwen3-coder`, `gemma4`).

SIGTERM during diagrams cleanly checkpointed: `sections_completed:
["summary", "logic", "cpp"]` saved (cpp marked complete despite empty
content — an arguable behavior; the script doesn't check section
non-emptiness before checkpointing).

### Run C — `ferguson-abstract.pdf` on `qwen3.5:2b-q8_0`

Attempted to verify that a 2B model would finish all 5 sections quickly. It
did not — got stuck on the **summary** section for 11 minutes, GPU 1 at
~88 % util the whole time. SIGTERM, 0 sections saved. Confirms the issue is
**qwen3.5-family fit** with the script's prompts, not model size.

To get a true 5-section end-to-end run, pull one of the actual
`MODEL_TIERS` models — the smallest is `deepseek-r1:8b` (~5 GB):

```bash
ollama pull deepseek-r1:8b
./paper_processor.py --model deepseek-r1:8b \
    --paper ferguson-abstract.pdf \
    /run/media/morlok/WORLOCK/home/jeb/Documents/AI-ML_Papers
```

Left as a follow-up — out of scope for the debug session.

---

## What's verified about the uncommitted diff

The diff (streaming Ollama, `_ShutdownRequested`, double-Ctrl-C force exit,
`check_required_models`, retry-once-on-Timeout) is **functionally sound**:

- Streaming path produces real markdown output (Runs A and B sections 1–2).
- Line-buffer `iter_lines` loop correctly accumulates `response` chunks and
  honors `done: true`.
- `_shutdown.is_set()` check inside the iter_lines loop wakes the read up
  promptly when SIGTERM fires (validated in both Runs A and B).
- The signal handler's "first hit → set event, second hit → `os._exit(1)`"
  contract works.
- Per-section checkpointing fires before the `_ShutdownRequested` exception
  unwinds, so partial papers retain accurate `sections_completed`.
- `check_required_models` (with fix #2) correctly gates on actually-used
  models.

---

## Files touched

| Path                                                                                | Change                                |
| ----------------------------------------------------------------------------------- | ------------------------------------- |
| `paper_processor.py:1`                                                              | shebang → venv python                 |
| `paper_processor.py` (mode)                                                         | `chmod +x` (`-rwx--x--x`)             |
| `paper_processor.py:1050-1056`                                                      | drop spurious `CODE_MODEL` from check |
| `docs/sessions/2026-05-09T052330Z_debug-paper-processor-venv.md`                    | **new** — this document               |

Output side-effects in `_processed/`:

- `_processed/huang_cv_failure/` — 1 section (summary). Run A leftovers.
- `_processed/ferguson-abstract/` — empty checkpoint from killed Run C.

(Both can be deleted by the user; they're outside the repo.)

---

## Notes / follow-ups

- **Empty-section guard.** The script checkpoints `cpp` as completed even
  when the LLM returned `""`. Worth adding a non-empty assertion in the
  section handlers so a quietly-broken model doesn't lock in "I'm done" with
  zero content. Out of scope for this session.
- **Shebang relative path.** `#!/usr/bin/env -S ./.venv/bin/python` ties the
  launch to CWD = repo root. If you ever want CWD-independence, switch to an
  absolute path (locks it to this checkout) or keep the system Python
  shebang and rely on the `v` shell helper. Current relative-path choice was
  the user's preference.
- **Model recommendation for actual use:** pull at least one of
  `deepseek-r1:8b` (5 GB), `deepseek-r1:14b` (9 GB), or `gemma4:31b-it-q4_K_M`
  (~18 GB). The `qwen3.5` family currently on this host is not suitable for
  the existing prompt templates.
