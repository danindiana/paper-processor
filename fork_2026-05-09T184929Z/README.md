# Paper Processor: 8.9GB Dual-GPU Residency Fork 🦞

This is a high-performance fork of the `paper_processor` pipeline, architected specifically for dual-GPU systems (e.g., RTX 3080 + RTX 3060) to achieve maximum throughput through **strict GPU pinning** and **multi-model residency**.

## Core Architecture: Strict GPU Pinning

Unlike standard Ollama setups that load-balance across available GPUs, this fork utilizes two independent Ollama backend instances to ensure models stay "hot-loaded" and never evict each other.

- **Backend 0 (Port 11434):** Pinned to **GPU 0** (RTX 3080, 10GB).
- **Backend 1 (Port 11435):** Pinned to **GPU 1** (RTX 3060, 12GB).

### Optimized Model Fitness Mapping

To maintain multi-model residency (keeping both a reasoning model and a code model in VRAM at all times), the fork uses GPU-aware model selection:

| Component | GPU 0 (RTX 3080, 10GB) | GPU 1 (RTX 3060, 12GB) |
| :--- | :--- | :--- |
| **Reasoning** | `deepseek-r1:8b` (5.2GB) | `deepseek-r1:8b` (5.2GB) |
| **Code (C++)** | `qwen3.5:4b` (3.4GB) | `ministral-3:8b` (6.0GB) |
| **Combined VRAM** | **8.6 GB** (Fits!) | **11.2 GB** (Fits!) |

## Features

- **Double Throughput:** Process two papers simultaneously with `--workers 2`.
- **Zero Eviction:** Reasoning and Code models are interleaved but never evicted because they co-exist in the assigned GPU's memory.
- **Context Capping:** Preserves residency on VRAM-constrained hardware:
  - **GPU 0 (3080):** Capped at **8,192** context window to fit dual models.
  - **GPU 1 (5080):** Full **32,768** context window enabled.
- **Parallel Chunking:** Long papers utilize `OLLAMA_NUM_PARALLEL` to process map-reduce chunks concurrently within a single GPU.
- **Modern C++:** Generates C++20/23 implementations of paper algorithms.
- **Neon Diagrams:** Automated Graphviz generation with a dark/neon aesthetic.

## Setup & Requirements

### 1. Dual Ollama Services
You must have two Ollama instances running.

**Primary (`ollama.service`):**
```ini
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="OLLAMA_NUM_PARALLEL=2"
```

**Secondary (`ollama-secondary.service`):**
```ini
Environment="OLLAMA_HOST=0.0.0.0:11435"
Environment="CUDA_VISIBLE_DEVICES=1"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
Environment="OLLAMA_NUM_PARALLEL=2"
```

### 2. Required Models
Ensure both instances have access to the optimized models:
```bash
ollama pull deepseek-r1:8b
ollama pull qwen3.5:4b
ollama pull ministral-3:8b
ollama pull qwen2.5:3b
```

## Usage

Run with 2 workers to activate both GPUs:
```bash
./paper_processor.py --workers 2
```

Reprocess specific sections across all papers:
```bash
./paper_processor.py --reprocess diagrams --workers 2
```

## Output Structure
```text
_processed/<paper_slug>/
├── metadata.json          # Model used, timestamps, and audit trail
├── 01_summary.md          # Comprehensive summary
├── 02_symbolic_logic.md   # Formal mathematical notation
├── 03_cpp_examples.md     # Modern C++20 implementations
├── 04_extras.md           # Critical analysis & open questions
└── diagrams/              # Neon-themed Graphviz SVG/DOT files
```
