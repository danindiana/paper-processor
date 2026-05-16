# Session: gpt-oss:20b output quality assessment

**Date:** 2026-05-14  
**Operator:** jeb  
**Host:** worlock (192.168.1.151)  
**Fork:** `fork_gptOSS_textonly_2026-05-14T205304Z`

## Paper processed

`019-3-034396-CT2578.pdf` — Aeroflex CT2578/CT2581 MIL-STD-1553B/1760 Remote
Terminal technical summary (Rev B, 3 Nov 1998). 22 pages → routes to
`xl_quality` = `gpt-oss:20b`. Reprocessed with `--reprocess all` to exercise
the new model tier.

## Section-by-section assessment

### 01 Summary — Excellent

- Correctly identified the document as a hardware datasheet, not a research
  paper, and adapted accordingly (no spurious "experimental setup" padding).
- Methodology table (Bus Interface / State Machine / Optional Feature Logic)
  accurately reflects the three-layer architecture.
- Timing specs (500 ns/word, 0.15 W/channel) and feature list pulled precisely
  from the source without hallucination.
- "Bottom Line" paragraph is concise and accurate.

### 02 Symbolic Logic — Impressive

- Formalizes the RT as a proper FSM: `R = (Q, Σ, δ, λ)` with
  `Q = {Idle, Recv, Trans, Mode, Err}`.
- Derives three substantive theorems: Status-Word Generation, Buffer Capacity
  (32-word overflow → Message-Error), and Timing Compliance.
- Algorithm pseudocode for BC→RT, RT→BC, RT↔RT, and both mode-code variants —
  all with loop invariants and O(n) complexity notes.
- PAC-learning analogy and information-theoretic view (entropy, mutual
  information, channel capacity) are creative and mathematically sound for a
  1553 hardware spec.

### 03 C++ Examples — Solid

- `CommandWord` and `StatusRegister` bit-field layouts match the actual 16-bit
  spec word exactly.
- `DataBuffer` is a correct 32-word circular buffer using `std::array`.
- Uses genuine C++20: `std::expected`, `std::optional`, `std::ranges`.
- `ManchesterEncoder` is a faithful behavioral model.
- Minor nit: some error paths return `std::unexpected` for valid-but-special
  cases (broadcast, wraparound) rather than routing them through the state
  machine — acceptable for a simulation sketch.

### 04 Extras — Strong

- Open-questions table is specific and actionable: buffer stress test under
  burst traffic, fault injection, temperature-dependent timing margins.
- Related-work connections are non-trivial: McAir ↔ CAN-FD bus-idle,
  open-drain status ↔ wired-OR safety patterns, checksum engine ↔ CRC-16
  precursor.
- Critical assessment is honest: correctly flags the absence of quantitative
  checksum error-detection capability and dated 5 V supply as real gaps.
- Pitch/critique balance is well-calibrated.

### 05 Diagrams — Clean render, one artifact

All 6 SVGs rendered without error. One pipeline artifact observed: the
diagrams directory contains **duplicate dot/svg pairs per concept** (two
naming conventions — `01_diagram_1___high_level_...` and
`01_high-level_...`). This is a pre-existing retry-slug issue in the parent
pipeline, not a model regression. 12 files instead of the expected 6; both
copies are valid DOT/SVG.

## Overall verdict

`gpt-oss:20b` produced output quality indistinguishable from the `gemma4`
tier on this document. No degradation detected across any section. The model
handled a non-typical input (a 1998 avionics hardware datasheet rather than
an ML paper) robustly, producing meaningful formalizations where a lesser
model might have fallen back to boilerplate.

The duplicate-diagram artifact is a known upstream issue, not introduced by
the model swap.

## VRAM observation

No OOM events. Ollama evicted models as expected between `gpt-oss:20b`
(xl_quality, ~12 GB) and `qwen3-coder:30b` (xl_code, ~17 GB) sections.
Freed ~6 GB headroom vs. the previous `gemma4:31b-it-q4_K_M` tier.
