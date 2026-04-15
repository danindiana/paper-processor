# paper-processor

A two-part pipeline for turning an AI/ML research-paper PDF corpus into
structured study dossiers using local LLMs:

- **`paper_processor.py`** — Python pipeline. For each PDF it produces a
  comprehensive summary, a formal-logic refactor, C++20/23 reference
  implementations of the key algorithms, six Graphviz diagrams (neon
  on black), and a critical-analysis doc. Metadata tracks progress so
  runs are resumable across interruptions.
- **`wizard/`** — [Ratatui](https://ratatui.rs/) TUI (`paper-wizard`)
  that explains the pipeline, scans your corpus, configures the run,
  launches the Python process, and streams its log with syntax colour.

Everything runs locally against [Ollama](https://ollama.com/); there is
no cloud dependency.

![neon-diagram-aesthetic](https://img.shields.io/badge/theme-neon--on--black-00FF41?style=flat-square)
![rust-ratatui](https://img.shields.io/badge/TUI-ratatui-FF00FF?style=flat-square)
![ollama](https://img.shields.io/badge/backend-ollama-00FFFF?style=flat-square)

## Output per paper

```
_processed/<subfolder>/<slug>/
├── 01_summary.md              # motivation → results → limitations
├── 02_symbolic_logic.md       # formal notation, theorems, complexity
├── 03_cpp_examples.md         # C++20/23 implementations, compilable
├── 04_extras.md               # open questions, critique, connections
├── diagrams/
│   ├── 01_<title>.dot + .svg  # architecture
│   ├── 02_<title>.dot + .svg  # data flow
│   ├── 03_<title>.dot + .svg  # algorithm flowchart
│   ├── 04_<title>.dot + .svg  # taxonomy
│   ├── 05_<title>.dot + .svg  # training loop
│   └── 06_<title>.dot + .svg  # vs prior art
└── metadata.json              # model, hash, sections completed
```

## Hardware targets

Written for a dual-GPU workstation (validated on RTX 3080 10 GB + RTX 3060
12 GB). Models auto-route by page count:

| Pages | Model | VRAM |
|---|---|---|
| ≤ 8 | `deepseek-r1:8b` | ~5 GB |
| ≤ 18 | `deepseek-r1:14b` | ~9 GB |
| > 18 | `gemma4:31b-it-q4_K_M` | ~18 GB dual-GPU |
| C++ stage | `qwen3-coder:30b` | ~17 GB dual-GPU |

Single-GPU setups: force `--model deepseek-r1:14b` for all stages.

## Quick start

```bash
# System + Python deps
bash setup_paper_processor.sh

# Pull the models you want (see table above)
ollama pull deepseek-r1:8b
ollama pull deepseek-r1:14b
ollama pull gemma4:31b-it-q4_K_M
ollama pull qwen3-coder:30b

# Run headless on a directory of PDFs (recursive)
python paper_processor.py ~/my_papers

# Single paper, forced small model
python paper_processor.py ~/my_papers --paper "attention.pdf" \
                                       --model deepseek-r1:14b

# Re-run just the diagram stage
python paper_processor.py ~/my_papers --reprocess diagrams

# Status of the whole corpus, no LLM calls
python paper_processor.py ~/my_papers --list
```

## TUI wizard

```bash
cd wizard
cargo build --release
sudo ln -sf "$PWD/target/release/paper-wizard" /usr/local/bin/paper-wizard
paper-wizard
```

Five tabs: **Overview · Scan · Config · Run · Help**. Streams the Python
subprocess output live with colour coding; `L` launches, `X` kills,
`Tab` cycles panels, `?` pops a key-ref overlay. Scan status is read
from each paper's `metadata.json` using the same slug logic as the
Python side, so partial runs are recognised and resumed correctly.

See [`wizard/README.md`](wizard/README.md) for full keybindings.

## CLI flags

```
python paper_processor.py [papers_dir]
    --backend {ollama,openclaw}   default ollama
    --model MODEL                 force one model for every stage
    --paper FILENAME              single paper (basename or rel path)
    --reprocess SECTION           summary|logic|cpp|diagrams|extras|all
    --workers N                   parallel papers (⚠ VRAM)
    --list                        show status table, exit
```

## Design notes

- **Resumable.** `metadata.json` records which sections finished. Re-running
  skips completed papers entirely and skips completed sections of partial
  papers. Safe to Ctrl-C anytime.
- **Recursive.** `papers_dir` is walked with `rglob`; the output tree
  mirrors the input subfolder structure so slugs can never collide.
- **Sliding-window chunking.** Papers longer than 12 pages are chunked
  with 2-page overlap, map-reduced per chunk, then condensed before the
  downstream prompts run on the distilled context.
- **Diagram aesthetic.** The LLM is explicitly instructed to emit neon
  accent colours on a black background; a post-processor injects
  `bgcolor=black` + default neon styles if the model forgets.
- **`num_gpu` intentionally unset.** Hard-forcing all layers on GPU causes
  OOM on 30B models at `num_ctx=32768` across a ≤22 GB VRAM pool; letting
  Ollama auto-schedule fixes it.

## Status

Built and validated on a single 4-page paper through all five sections +
six SVG diagrams. Runtime budget for a 5,000-paper corpus at average
~20-40 min/paper is measured in weeks, not hours — plan in batches.

## Licence

MIT
