#!/usr/bin/env -S ../.venv/bin/python
"""
paper_processor.py — OpenClaw / Ollama AI-ML Paper Processing Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each PDF in the target directory, produces:
  01_summary.md          — Comprehensive paper summary
  02_symbolic_logic.md   — Core insights in formal symbolic logic
  03_cpp_examples.md     — C++20/23 implementations of key algorithms
  diagrams/              — 6+ Graphviz DOT + rendered SVG (neon/black)
  04_extras.md           — Open questions, connections, critical assessment
  metadata.json          — Audit trail (model, hash, timestamps, strategy)

Usage:
  python paper_processor.py                               # all papers, ollama backend
  python paper_processor.py --backend openclaw            # use OpenClaw agent CLI
  python paper_processor.py --model deepseek-r1:14b      # force a model
  python paper_processor.py --paper "attention.pdf"      # single paper
  python paper_processor.py --list                        # show status table
  python paper_processor.py --reprocess diagrams         # redo one section
  python paper_processor.py --workers 2                  # parallel (VRAM permitting)
  python paper_processor.py --override                   # evict loaded models, restart if stuck
"""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# ── Third-party imports ────────────────────────────────────────────────────
try:
    import fitz  # pymupdf
except ImportError:
    sys.exit("❌  pymupdf not installed.\n    Fix: pip install pymupdf --break-system-packages")

try:
    import requests
except ImportError:
    sys.exit("❌  requests not installed.\n    Fix: pip install requests --break-system-packages")


# ══════════════════════════════════════════════════════════════════════════════
# GRACEFUL SHUTDOWN
# ══════════════════════════════════════════════════════════════════════════════
_shutdown = threading.Event()


def _install_signal_handlers() -> None:
    def _handler(signum, frame):
        if _shutdown.is_set():
            print("\n  🚨  Forced exit requested — stopping immediately!")
            os._exit(1)
        print("\n\n  ⚡  Shutdown requested — finishing current section then stopping …")
        print("      (Press Ctrl+C again to force exit immediately)")
        _shutdown.set()
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


class _ShutdownRequested(Exception):
    """Raised when _shutdown fires during a blocking LLM call or map-reduce loop."""


# ══════════════════════════════════════════════════════════════════════════════
# HARDWARE CONTEXT
# RTX 3080  → 10 240 MiB VRAM   (GPU 0, display attached)
# RTX 5080  → 16 376 MiB VRAM   (GPU 1)
# Total     →  ~22 GB  (Ollama auto-spans with CUDA_VISIBLE_DEVICES or its own scheduler)
# RAM       → 128 GB   (no OOM concern for CPU offload)
# ══════════════════════════════════════════════════════════════════════════════

# Models and mapping are now GPU-aware (see GPU_OPTIMIZED_MODELS)

# ══════════════════════════════════════════════════════════════════════════════
# OLLAMA GPU PROVISIONER  (--override mode)
# ══════════════════════════════════════════════════════════════════════════════

def _ollama_get_loaded() -> List[str]:
    """Return names of models currently resident in VRAM via /api/ps."""
    url = get_gpu_url()
    try:
        r = requests.get(f"{url}/api/ps", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def _ollama_evict(model: str) -> bool:
    """Ask Ollama to unload a model immediately (keep_alive=0)."""
    url = get_gpu_url()
    try:
        r = requests.post(
            f"{url}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=20,
        )
        return r.status_code == 200
    except Exception:
        return False


def _ollama_wait_clean(timeout: int = 30, interval: float = 2.0) -> bool:
    """Poll /api/ps until no models are loaded or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _ollama_get_loaded():
            return True
        time.sleep(interval)
    return False


def _ollama_restart_service() -> bool:
    """Restart the ollama systemd service and wait for it to come back online."""
    print("  🔄  Restarting ollama service …")
    url = get_gpu_url()
    try:
        # Note: we restart the main service only. Secondary service restart
        # is omitted here for simplicity as it's less likely to hang.
        r = subprocess.run(
            ["sudo", "systemctl", "restart", "ollama"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            print(f"  ⚠️   systemctl restart failed: {r.stderr.strip()}")
            return False
    except Exception as exc:
        print(f"  ❌  Could not restart ollama: {exc}")
        return False

    print("  ⏳  Waiting for Ollama to come back up …", end="", flush=True)
    deadline = time.time() + 45
    while time.time() < deadline:
        time.sleep(2)
        try:
            r = requests.get(f"{url}/api/tags", timeout=3)
            if r.status_code == 200:
                print(" up!")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
    print(" timed out")
    return False


def provision_ollama(verbose: bool = False) -> bool:
    """
    Aggressively free GPU VRAM before processing starts.
    """
    print("\n  🎯  Override: provisioning Ollama for exclusive GPU use …")

    loaded = _ollama_get_loaded()
    if not loaded:
        print("  ✓   No models currently loaded — GPU is free")
        return True

    # Level 1: graceful eviction via keep_alive=0
    print(f"  ⚡  Evicting {len(loaded)} loaded model(s): {', '.join(loaded)}")
    for model in loaded:
        print(f"       → {model} …", end="", flush=True)
        ok = _ollama_evict(model)
        print(" ✓" if ok else " ✗")

    if _ollama_wait_clean(timeout=20):
        print("  ✅  All models evicted — GPU is free\n")
        return True

    # Level 2: force a service restart
    remaining = _ollama_get_loaded()
    print(f"  ⚠️   {len(remaining)} model(s) still loaded — escalating to service restart")
    if not _ollama_restart_service():
        print("  ⚠️   Service restart failed — proceeding anyway\n")
        return False

    if _ollama_wait_clean(timeout=15):
        print("  ✅  Ollama restarted clean\n")
        return True

    print("  ⚠️   Could not confirm clean VRAM — proceeding anyway\n")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════
PROMPTS = {
    "summary": textwrap.dedent("""\
        Write a comprehensive, well-structured summary of this AI/ML research paper.
        Your summary must cover ALL of the following:
          1. Motivation & Problem Statement — what gap does this address?
          2. Core Methodology — how does the approach work at a high level?
          3. Key Contributions — what is genuinely novel?
          4. Experimental Setup & Results — benchmarks, datasets, metrics, numbers
          5. Limitations & Failure Modes — where does the approach break down?
          6. Significance — how does this advance the field?
        Be thorough and precise. Use section headers."""),

    "logic": textwrap.dedent("""\
        Refactor the paper's core insights using rigorous symbolic logic and formal notation.
        Structure your response as:

        ## 1. Core Definitions & Notation
        Define all entities, sets, and functions with formal notation (∈, ⊂, →, ℝⁿ, etc.)

        ## 2. Key Theorems & Propositions
        State the paper's central claims as formal propositions with ∀, ∃, →, ↔ quantifiers.

        ## 3. Algorithm Formalisation
        Express each major algorithm using pseudocode with mathematical notation,
        loop invariants, and complexity bounds (O, Ω, Θ).

        ## 4. Optimality & Convergence Conditions
        State any convergence theorems, loss landscape properties, or PAC-learning bounds.

        ## 5. Information-Theoretic View
        Express the core learning objective using entropy H(·), KL divergence, mutual information I(·;·) where applicable."""),

    "cpp": textwrap.dedent("""\
        Refactor the paper's core insights using well-crafted C++ code examples.
        Requirements:
          - Use modern C++20 / C++23 (concepts, ranges, coroutines, std::expected, etc.)
          - Implement the key algorithms and data structures described in the paper
          - Each code block must open with a comment block citing the relevant paper section/equation
          - Provide at least 3 self-contained, compilable examples
          - Include a main() that exercises each implementation with sample data
          - Prefer STL containers and algorithms; avoid raw owning pointers
          - Show template metaprogramming or concept constraints where they model the paper's abstractions
          - Add inline comments explaining the mapping from math → code"""),

    "extras": textwrap.dedent("""\
        Provide deep additional analysis beyond what the paper itself claims:

        ## 1. Open Questions
        What does this paper leave unresolved? What follow-up experiments are obviously needed?

        ## 2. Related Work & Connections
        How does this relate to other landmark AI/ML papers? What does it supersede?
        What does it complement? Are there surprising connections to other subfields?

        ## 3. Practical Deployment Considerations
        Real-world tradeoffs: latency, memory, data requirements, failure modes in production.

        ## 4. Critical Assessment
        Evaluate the paper's claims critically:
          - Is the experimental setup fair and reproducible?
          - Are there cherry-picked baselines?
          - Do the ablations actually support the claimed conclusions?

        ## 5. Surprising or Underappreciated Insights
        What does the paper imply but not say explicitly? What would a careful reader notice
        that a casual reader would miss?

        ## 6. One-Paragraph Pitch & One-Paragraph Critique
        Steelman the paper in one paragraph. Then write the strongest possible critique in one paragraph."""),
}

# Diagram generation prompt — model must return delimited DOT blocks
DIAGRAM_PROMPT = textwrap.dedent("""\
    Generate exactly 6 Graphviz DOT diagrams that illuminate this AI/ML paper from 6 different angles:

      Diagram 1 — High-Level Architecture / System Overview
      Diagram 2 — Data Flow & Processing Pipeline
      Diagram 3 — Core Algorithm as a Flowchart
      Diagram 4 — Concept Taxonomy / Knowledge Hierarchy
      Diagram 5 — Training Loop / Optimisation Dynamics
      Diagram 6 — Comparison vs Prior Art (or Ablation Structure)

    ══ MANDATORY VISUAL STYLE (apply to EVERY diagram) ══
      graph-level:  bgcolor="black"
      node default: style=filled, fillcolor="#0a0a0a", fontname="Courier New", fontsize=11
      NEON accent hex colours — ALWAYS quoted in attributes:
        "#00FF41"  "#FF00FF"  "#00FFFF"  "#FF6600"  "#FFFF00"
        "#FF0055"  "#7FFF00"  "#0080FF"  "#DA70FF"
      Example:  node [color="#00FF41"]  — NEVER write  color=Electric Green
      Edges: penwidth=2.0, vary neon colour per diagram
      Graph titles: label= + labelloc=t with a bright fontcolor
      Mix rankdir=LR and rankdir=TB between diagrams.
      Node IDs: letters/digits/underscores ONLY — no spaces, ?, /, etc.
      Use quoted labels for display text:  CONVERGE [label="Converge?"]

    ══ OUTPUT FORMAT — strictly follow this delimiter pattern ══
    ===DIAGRAM_START: <Descriptive Title for Diagram N>===
    digraph G {
      // ... full valid DOT source ...
    }
    ===DIAGRAM_END===

    Output ONLY the 6 delimited DOT blocks. No prose before, between, or after.""")


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Metadata:
    paper_name: str
    pdf_path: str
    page_count: int
    chunk_strategy: str
    model_used: str
    code_model: str
    processed_at: str
    paper_hash: str
    sections_completed: List[str] = field(default_factory=list)

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Optional["Metadata"]:
        try:
            data = json.loads(path.read_text())
            return cls(**data)
        except Exception:
            return None


ALL_SECTIONS = {"summary", "logic", "cpp", "diagrams", "extras"}


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND  (Ollama direct API  or  OpenClaw agent CLI)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# MULTI-GPU & MULTI-BACKEND MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
_gpu_lock = threading.Lock()
_gpu_available = [0, 1]  # GPU 0 (3080, port 11434) and GPU 1 (3060, port 11435)
_thread_to_gpu = {}

# GPU-specific optimized model mapping to maximize fitness and residency
# Goal: keep BOTH reasoning and code models hot-loaded on each GPU.
GPU_OPTIMIZED_MODELS = {
    0: {  # RTX 3080 (10 GB)
        "reason": "deepseek-r1:8b",      # 5.2 GB
        "code":   "qwen2.5-coder:7b",    # 4.4 GB (Total: 9.6 GB — tight; 8k ctx cap keeps actual VRAM ~9.5 GB)
        "fast":   "qwen2.5:3b",          # 1.9 GB
    },
    1: {  # RTX 5080 (16 GB)
        "reason": "deepseek-r1:8b",      # 5.2 GB
        "code":   "ministral-3:8b",      # 6.0 GB (Total: 11.2 GB — Fits!)
        "fast":   "qwen2.5:3b",          # 1.9 GB
    }
}

BACKEND_URLS = {
    0: "http://localhost:11434",
    1: "http://localhost:11435",
}

def get_assigned_gpu() -> int:
    """Return the GPU ID assigned to the current thread."""
    return _thread_to_gpu.get(threading.get_ident(), 0)

def get_gpu_url() -> str:
    """Return the Ollama URL for the assigned GPU."""
    return BACKEND_URLS.get(get_assigned_gpu(), "http://localhost:11434")

def get_gpu_model(tier: str) -> str:
    """Return the optimized model for the current GPU and tier."""
    gpu_id = get_assigned_gpu()
    models = GPU_OPTIMIZED_MODELS.get(gpu_id, GPU_OPTIMIZED_MODELS[0])
    
    if tier in ("xl_quality", "xl_reason", "mid_reason", "single"):
        return models["reason"]
    if tier in ("xl_code", "mid_code", "single_code"):
        return models["code"]
    return models.get(tier, models["reason"])


class Backend:
    """
    Thin abstraction over LLM backends.
    """

    def __init__(self, name: str, default_model: str):
        self.name          = name
        self.default_model = default_model

    def call(
        self,
        prompt: str,
        model: Optional[str] = None,
        ctx_tokens: int = 32768,
    ) -> str:
        m = model or self.default_model
        gpu_id = get_assigned_gpu()
        
        # GPU-specific context window limits to preserve VRAM residency
        # GPU 0 (3080, 10GB) capped at 8k to fit reasoning + code models
        # GPU 1 (5080, 16GB) allowed full 32k
        effective_ctx = ctx_tokens
        if gpu_id == 0:
            effective_ctx = min(ctx_tokens, 8192)
            
        if self.name == "ollama":
            return self._call_ollama(prompt, m, effective_ctx)
        return self._call_openclaw(prompt, m)

    # ── Ollama ────────────────────────────────────────────────────────────
    def _call_ollama(self, prompt: str, model: str, ctx: int) -> str:
        url = f"{get_gpu_url()}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_ctx":        ctx,
                "temperature":    0.20,
                "top_p":          0.90,
                "repeat_penalty": 1.10,
            },
        }
        
        for attempt in range(1, 3):
            if _shutdown.is_set():
                return ""

            try:
                r = requests.post(url, json=payload, timeout=1200, stream=True)
                if not r.ok:
                    try:
                        body = r.json().get("error", r.text[:200])
                    except Exception:
                        body = r.text[:200]
                    raise RuntimeError(f"Ollama HTTP {r.status_code} at {url} (model={model}): {body}")

                full_response = []
                for line in r.iter_lines():
                    if _shutdown.is_set():
                        r.close()
                        raise _ShutdownRequested()
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            full_response.append(data["response"])
                        if data.get("done"):
                            break
                return "".join(full_response).strip()

            except requests.exceptions.ConnectionError:
                sys.exit(
                    f"❌  Cannot reach Ollama at {url}\n"
                    "    Is the service running?"
                )
            except requests.exceptions.Timeout as exc:
                if attempt == 1 and not _shutdown.is_set():
                    print(f"     ⏱  Ollama timed out at {url} — retrying (model={model}) …")
                    continue
                raise RuntimeError(f"Ollama error (model={model}): {exc}") from exc
            except (_ShutdownRequested, RuntimeError):
                raise
            except Exception as exc:
                raise RuntimeError(f"Ollama error (model={model}): {exc}") from exc

    # ── OpenClaw ──────────────────────────────────────────────────────────
    def _call_openclaw(self, prompt: str, model: str) -> str:
        env = os.environ.copy()
        gpu_id = get_assigned_gpu()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        cmd = ["openclaw", "agent", "--agent", "main", "--message", prompt]

        try:
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            ) as proc:
                deadline = time.time() + 1200
                while proc.poll() is None:
                    if _shutdown.is_set():
                        proc.terminate()
                        return ""
                    if time.time() > deadline:
                        proc.kill()
                        raise RuntimeError("OpenClaw agent timed out")
                    time.sleep(1.0)

                stdout, stderr = proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(f"openclaw agent exited {proc.returncode}")
                return stdout.strip()
        except Exception as exc:
            raise RuntimeError(f"OpenClaw error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# PDF  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def extract_pages(pdf_path: Path) -> List[str]:
    """Return list of text strings, one per page."""
    doc   = fitz.open(str(pdf_path))
    pages = [page.get_text("text") for page in doc]
    doc.close()
    # Drop blank pages
    return [p for p in pages if p.strip()]


def select_model(page_count: int, user_override: Optional[str]) -> str:
    if user_override:
        return user_override
    tier = "fast" if page_count <= 10 else "xl_quality"
    return get_gpu_model(tier)


def build_chunks(pages: List[str], window: int = 12, overlap: int = 2) -> List[str]:
    """Sliding-window chunking with overlap to preserve cross-page context."""
    if len(pages) <= window:
        return ["\n\n".join(pages)]
    chunks, i = [], 0
    while i < len(pages):
        end = min(i + window, len(pages))
        chunks.append("\n\n".join(pages[i:end]))
        if end == len(pages):
            break
        i += window - overlap
    return chunks


def map_reduce_chunks(
    chunks: List[str],
    backend: Backend,
    model: str,
) -> str:
    """Summarise each chunk, then return joined summaries as condensed context."""
    if len(chunks) == 1:
        return chunks[0]

    num_parallel = int(os.environ.get("OLLAMA_NUM_PARALLEL", "2"))
    print(f"      ↳ Map-reduce: {len(chunks)} chunks (parallel={num_parallel}) …")

    def _process_chunk(idx_chunk: Tuple[int, str]) -> Tuple[int, str]:
        idx, chunk = idx_chunk
        if _shutdown.is_set():
            return idx, ""
        prompt = (
            f"You are reading chunk {idx} of {len(chunks)} of an AI/ML research paper. "
            "Summarise this chunk concisely, preserving all technical details, "
            "equations, and experimental results:\n\n" + chunk[:18000]
        )
        summary = backend.call(prompt, model, ctx_tokens=16384)
        print(f"        chunk {idx}/{len(chunks)} ✓")
        return idx, f"### Chunk {idx}/{len(chunks)}\n{summary}"

    # Process chunks in parallel using ThreadPoolExecutor
    indexed_chunks = list(enumerate(chunks, 1))
    results: List[Tuple[int, str]] = []

    with ThreadPoolExecutor(max_workers=num_parallel) as executor:
        futures = {executor.submit(_process_chunk, ic): ic for ic in indexed_chunks}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"        chunk failed: {exc}")

    if _shutdown.is_set():
        raise _ShutdownRequested()

    # Ensure results are in the original order
    results.sort(key=lambda x: x[0])
    return "\n\n---\n\n".join(res for idx, res in results if res)


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM  PARSING  +  RENDERING
# ══════════════════════════════════════════════════════════════════════════════
_DELIM_RE = re.compile(
    r"===DIAGRAM_START:\s*(.+?)===\s*(.*?)===DIAGRAM_END[^\n]*",
    re.DOTALL | re.IGNORECASE,
)
# Fallback: bare ``` or ```dot fences
_FENCE_RE = re.compile(
    r"```(?:dot|graphviz)?\s*\n?((?:digraph|graph)\b.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def parse_diagrams(raw: str) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []

    for m in _DELIM_RE.finditer(raw):
        title = m.group(1).strip()
        dot   = m.group(2).strip()
        if dot and ("digraph" in dot.lower() or "graph" in dot.lower()):
            results.append((title, dot))

    if not results:  # fallback — try fenced blocks
        for idx, m in enumerate(_FENCE_RE.finditer(raw), 1):
            results.append((f"diagram_{idx:02d}", m.group(1).strip()))

    return results


def ensure_neon_black(dot_src: str) -> str:
    """
    Inject bgcolor=black and a default neon node style if the LLM forgot.
    Non-destructive: skips injection if already present.
    """
    if "bgcolor" not in dot_src:
        dot_src = re.sub(
            r"((?:di)?graph\s+\w*\s*\{)",
            r'\1\n  graph [bgcolor="black" fontcolor="#00FF41" fontname="Courier New"];'
            r'\n  node  [style=filled fillcolor="#0a0a0a" color="#00FF41" fontcolor="#00FF41" fontname="Courier New"];'
            r'\n  edge  [color="#FF00FF" penwidth=2.0];',
            dot_src,
            count=1,
        )
    return dot_src


_COLOR_NAMES = {
    "Electric Green": "#00FF41", "Hot Magenta": "#FF00FF",
    "Cyan":           "#00FFFF", "Neon Orange":  "#FF6600",
    "Volt Yellow":    "#FFFF00", "Hot Pink":     "#FF0055",
    "Chartreuse":     "#7FFF00", "Electric Blue":"#0080FF",
    "Lavender":       "#DA70FF",
}


def sanitize_dot(dot_src: str) -> str:
    """Replace bare LLM color names with hex codes and strip any content after the closing brace."""
    for name, hex_code in _COLOR_NAMES.items():
        dot_src = re.sub(
            rf'(=\s*){re.escape(name)}(?=[;\]\s,\n]|$)',
            rf'\1"{hex_code}"',
            dot_src,
            flags=re.IGNORECASE,
        )
    # Trim any raw LLM delimiter text that leaked past the closing brace
    last_brace = dot_src.rfind("}")
    if last_brace != -1:
        dot_src = dot_src[: last_brace + 1]
    return dot_src


def render_dot(dot_src: str, out_svg: Path) -> bool:
    try:
        r = subprocess.run(
            ["dot", "-Tsvg", "-o", str(out_svg)],
            input=dot_src,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if r.returncode != 0 and r.stderr.strip():
            print(f"      ⚠️  dot error: {r.stderr.strip()[:200]}")
        return r.returncode == 0
    except FileNotFoundError:
        print("      ⚠️  graphviz `dot` not found — SVGs will not be rendered")
        print("          Fix: sudo apt install graphviz")
        return False
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# PAPER  PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════
class PaperProcessor:
    def __init__(
        self,
        papers_dir: Path,
        backend: Backend,
        forced_model: Optional[str] = None,
        reprocess: Optional[str] = None,
        verbose: bool = False,
    ):
        self.papers_dir   = papers_dir
        # Isolation Fix: Store output locally within the fork directory
        self.out_root     = Path(__file__).parent / "_processed"
        self.backend      = backend
        self.forced_model = forced_model
        self.reprocess    = reprocess  # section name or "all"
        self.verbose      = verbose
        self.out_root.mkdir(exist_ok=True)

    # ── Utilities ─────────────────────────────────────────────────────────
    def _paper_dir(self, pdf: Path) -> Path:
        # Mirror the input tree under _processed/ so subfolder structure is preserved
        # and slug collisions across subdirs are impossible.
        try:
            rel_parent = pdf.parent.relative_to(self.papers_dir)
        except ValueError:
            rel_parent = Path("")
        parent_slug = Path(*[
            re.sub(r"[^\w\-]", "_", part).lower().strip("_")
            for part in rel_parent.parts
        ]) if rel_parent.parts else Path("")
        stem_slug = re.sub(r"[^\w\-]", "_", pdf.stem)[:64].lower().strip("_")
        d = self.out_root / parent_slug / stem_slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "diagrams").mkdir(exist_ok=True)
        return d

    def _should_run(self, section: str, completed: List[str]) -> bool:
        if self.reprocess in (section, "all"):
            return True
        return section not in completed

    def _write_md(self, path: Path, heading: str, body: str):
        path.write_text(f"# {heading}\n\n{body}\n", encoding="utf-8")

    def _tag_prompt(self, task_prompt: str, context: str) -> str:
        """Wrap paper context in XML tags for cleaner prompt structure."""
        return f"{task_prompt}\n\n<paper>\n{context}\n</paper>"

    def _save_meta(
        self,
        meta_path: Path,
        pdf_path: Path,
        page_count: int,
        strategy: str,
        model: str,
        code_model: str,
        paper_hash: str,
        completed: List[str],
    ) -> None:
        Metadata(
            paper_name        = pdf_path.name,
            pdf_path          = str(pdf_path),
            page_count        = page_count,
            chunk_strategy    = strategy,
            model_used        = model,
            code_model        = code_model,
            processed_at      = time.strftime("%Y-%m-%dT%H:%M:%S"),
            paper_hash        = paper_hash,
            sections_completed= completed,
        ).save(meta_path)

    # ── Main entry ────────────────────────────────────────────────────────
    def process(self, pdf_path: Path) -> None:
        paper_dir = self._paper_dir(pdf_path)
        meta_path = paper_dir / "metadata.json"
        meta      = Metadata.load(meta_path)

        # Skip if fully complete and no reprocess requested
        if (
            meta is not None
            and ALL_SECTIONS.issubset(set(meta.sections_completed))
            and self.reprocess is None
        ):
            print(f"  ⏭   {pdf_path.name}  (all sections complete)")
            return

        print(f"\n{'─'*64}")
        print(f"  📄  {pdf_path.name}")

        # ── Extract & chunk ───────────────────────────────────────────────
        pages      = extract_pages(pdf_path)
        page_count = len(pages)
        model      = select_model(page_count, self.forced_model)
        code_model = self.forced_model or get_gpu_model("xl_code")

        chunks     = build_chunks(pages)
        strategy   = (
            f"single-pass ({page_count} pages)"
            if len(chunks) == 1
            else f"sliding-window ({len(chunks)} chunks, {page_count} pages)"
        )

        print(f"     pages={page_count}  chunks={len(chunks)}")
        print(f"     model={model}")
        print(f"     code_model={code_model}")
        print(f"     strategy={strategy}")

        # Compute once — used for checkpoints throughout
        paper_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]
        completed: List[str] = list(meta.sections_completed) if meta else []

        def _checkpoint():
            self._save_meta(meta_path, pdf_path, page_count, strategy,
                            model, code_model, paper_hash, completed)

        def _shutting_down() -> bool:
            if _shutdown.is_set():
                _checkpoint()
                print(f"     ⚡  Stopped — {len(completed)} section(s) saved")
                return True
            return False

        # Condense multi-chunk papers to a manageable context string
        try:
            context = map_reduce_chunks(chunks, self.backend, model)
        except _ShutdownRequested:
            _checkpoint()
            print(f"     ⚡  Stopped during map-reduce — {len(completed)} section(s) saved")
            return

        # Cap to ~90k chars (~22k tokens) — safe for 32k-ctx models
        capped = context[:90_000]

        # ── 1. Summary ────────────────────────────────────────────────────
        if self._should_run("summary", completed):
            if _shutting_down():
                return
            print("     📝  Summary …")
            try:
                out = self.backend.call(self._tag_prompt(PROMPTS["summary"], capped), model)
            except _ShutdownRequested:
                _checkpoint()
                print(f"     ⚡  Interrupted mid-section — {len(completed)} section(s) saved")
                return
            self._write_md(paper_dir / "01_summary.md", "Summary", out)
            if "summary" not in completed:
                completed.append("summary")
            _checkpoint()
            print("         ✓")

        # ── 2. Symbolic Logic ─────────────────────────────────────────────
        if self._should_run("logic", completed):
            if _shutting_down():
                return
            print("     🔣  Symbolic logic …")
            try:
                out = self.backend.call(self._tag_prompt(PROMPTS["logic"], capped), model)
            except _ShutdownRequested:
                _checkpoint()
                print(f"     ⚡  Interrupted mid-section — {len(completed)} section(s) saved")
                return
            self._write_md(paper_dir / "02_symbolic_logic.md", "Symbolic Logic Formulation", out)
            if "logic" not in completed:
                completed.append("logic")
            _checkpoint()
            print("         ✓")

        # ── 3. C++ Examples ───────────────────────────────────────────────
        if self._should_run("cpp", completed):
            if _shutting_down():
                return
            print(f"     💻  C++ examples  (model: {code_model}) …")
            try:
                out = self.backend.call(self._tag_prompt(PROMPTS["cpp"], capped), code_model)
            except _ShutdownRequested:
                _checkpoint()
                print(f"     ⚡  Interrupted mid-section — {len(completed)} section(s) saved")
                return
            self._write_md(paper_dir / "03_cpp_examples.md", "C++ Implementation Examples", out)
            if "cpp" not in completed:
                completed.append("cpp")
            _checkpoint()
            print("         ✓")

        # ── 4. Graphviz Diagrams ──────────────────────────────────────────
        if self._should_run("diagrams", completed):
            if _shutting_down():
                return
            print("     📊  Graphviz diagrams …")
            try:
                raw = self.backend.call(
                    self._tag_prompt(DIAGRAM_PROMPT, capped[:60_000]),
                    code_model,  # DOT generation is a structured-output/code task
                    ctx_tokens=32768,
                )
            except _ShutdownRequested:
                _checkpoint()
                print(f"     ⚡  Interrupted mid-section — {len(completed)} section(s) saved")
                return
            diagrams = parse_diagrams(raw)

            if not diagrams:
                # Save raw output so user can inspect / manually extract DOT
                raw_out = paper_dir / "diagrams" / "_raw_llm_output.txt"
                raw_out.write_text(raw, encoding="utf-8")
                print(
                    f"     ⚠️   No diagrams parsed from LLM output.\n"
                    f"          Raw output saved → {raw_out}\n"
                    f"          Tip: re-run with --reprocess diagrams after inspecting output."
                )
            else:
                for idx, (title, dot_src) in enumerate(diagrams, 1):
                    dot_src  = sanitize_dot(ensure_neon_black(dot_src))
                    safe     = re.sub(r"[^\w\-]", "_", title)[:40].lower().strip("_")
                    dot_path = paper_dir / "diagrams" / f"{idx:02d}_{safe}.dot"
                    svg_path = paper_dir / "diagrams" / f"{idx:02d}_{safe}.svg"
                    dot_path.write_text(dot_src, encoding="utf-8")
                    ok     = render_dot(dot_src, svg_path)
                    status = "✓" if ok else "✗ (dot saved, SVG render failed)"
                    print(f"       {idx}. {title:<45} {status}")

            if "diagrams" not in completed:
                completed.append("diagrams")
            _checkpoint()

        # ── 5. Extras ─────────────────────────────────────────────────────
        if self._should_run("extras", completed):
            if _shutting_down():
                return
            print("     💡  Extras / critical analysis …")
            try:
                out = self.backend.call(self._tag_prompt(PROMPTS["extras"], capped), model)
            except _ShutdownRequested:
                _checkpoint()
                print(f"     ⚡  Interrupted mid-section — {len(completed)} section(s) saved")
                return
            self._write_md(paper_dir / "04_extras.md", "Additional Insights", out)
            if "extras" not in completed:
                completed.append("extras")
            _checkpoint()
            print("         ✓")

        # ── Write / update metadata ────────────────────────────────────────
        _checkpoint()
        print(f"     ✅  Output → {paper_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def list_status(papers_dir: Path):
    processed_dir = papers_dir / "_processed"
    pdfs = sorted(p for p in papers_dir.rglob("*.pdf") if "_processed" not in p.parts)
    if not pdfs:
        print(f"  No PDFs found in {papers_dir}")
        return

    print(f"\n  {'Paper':<60}  {'Status'}")
    print(f"  {'─'*60}  {'─'*30}")
    for pdf in pdfs:
        try:
            rel_parent = pdf.parent.relative_to(papers_dir)
        except ValueError:
            rel_parent = Path("")
        parent_slug = Path(*[
            re.sub(r"[^\w\-]", "_", part).lower().strip("_")
            for part in rel_parent.parts
        ]) if rel_parent.parts else Path("")
        stem_slug = re.sub(r"[^\w\-]", "_", pdf.stem)[:64].lower().strip("_")
        meta_path = processed_dir / parent_slug / stem_slug / "metadata.json"
        meta      = Metadata.load(meta_path) if meta_path.exists() else None

        if meta is None:
            status = "⬜  not started"
        elif ALL_SECTIONS.issubset(set(meta.sections_completed)):
            status = f"✅  complete  [{meta.model_used}]"
        else:
            missing = ALL_SECTIONS - set(meta.sections_completed)
            status  = f"⚠️   partial — missing: {', '.join(sorted(missing))}"

        rel = pdf.relative_to(papers_dir).as_posix()
        name = rel[:58] + "…" if len(rel) > 58 else rel
        print(f"  {name:<60}  {status}")
    print()


def health_check_ollama():
    for gpu_id, url in BACKEND_URLS.items():
        try:
            r = requests.get(f"{url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"  🟢  Ollama reachable at {url} (GPU {gpu_id})")
        except Exception as exc:
            print(f"  ❌  Cannot reach Ollama at {url}: {exc}")
            return False
    return True


def check_required_models(models: List[str]) -> None:
    """Exit with pull instructions if any model is absent from Ollama's local library."""
    url = BACKEND_URLS[0]
    try:
        r = requests.get(f"{url}/api/tags", timeout=10)
        r.raise_for_status()
        available = {m["name"] for m in r.json().get("models", [])}
    except Exception:
        return
    missing = [m for m in models if m not in available]
    if missing:
        sys.exit(
            "❌  Required models not found locally — pull them first to avoid "
            "auto-download timeouts:\n"
            + "\n".join(f"    ollama pull {m}" for m in missing)
        )


def prewarm_models() -> None:
    """Send a 1-token ping to each model on each GPU to force load before first paper."""
    print("  🔥  Pre-warming models …")
    for gpu_id, url in BACKEND_URLS.items():
        models = GPU_OPTIMIZED_MODELS.get(gpu_id, GPU_OPTIMIZED_MODELS[0])
        for tier in ("reason", "code"):
            model = models[tier]
            try:
                payload = {
                    "model": model,
                    "prompt": "hi",
                    "stream": False,
                    "options": {"num_ctx": 512, "num_predict": 1},
                }
                r = requests.post(f"{url}/api/generate", json=payload, timeout=120)
                status = "✓" if r.ok else f"✗ HTTP {r.status_code}"
            except Exception as exc:
                status = f"✗ {exc}"
            print(f"     GPU {gpu_id}  {model:<32} {status}")


def health_check_openclaw():
    try:
        r = subprocess.run(
            ["openclaw", "health"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            print("  🟢  OpenClaw gateway healthy")
            return True
        print(f"  ⚠️   openclaw health returned {r.returncode}: {r.stderr.strip()}")
        return False
    except FileNotFoundError:
        print("  ❌  `openclaw` not found in PATH")
        return False


def main():
    ap = argparse.ArgumentParser(
        prog="paper_processor",
        description="🦞  OpenClaw / Ollama  AI-ML Paper Processing Pipeline (8.9GB Fork)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Output tree per paper:
              _processed/<slug>/
                metadata.json
                01_summary.md
                02_symbolic_logic.md
                03_cpp_examples.md
                04_extras.md
                diagrams/
                  01_<title>.dot
                  01_<title>.svg
                  …  (6+ diagrams)

            Model auto-selection (Optimized for Multi-GPU Residency):
              GPU 0 (RTX 3080, 10GB, 8k ctx):
                Reasoning/Summary/Logic/Extras: deepseek-r1:8b (5.2GB)
                Code/Diagrams:                  qwen2.5-coder:7b (4.4GB)
              GPU 1 (RTX 5080, 16GB, 32k ctx):
                Reasoning/Summary/Logic/Extras: deepseek-r1:8b (5.2GB)
                Code/Diagrams:                  ministral-3:8b (6.0GB)
              Routing: papers >20 pages are preferred to GPU 1 (larger context).
        """),
    )
    ap.add_argument(
        "papers_dir", nargs="?",
        default="/home/jeb/Documents/AI-ML_Papers",
        help="Directory containing PDF papers (default: ~/Documents/AI-ML_Papers)",
    )
    ap.add_argument(
        "--backend", choices=["ollama", "openclaw"], default="ollama",
        help="LLM backend: 'ollama' (direct API) or 'openclaw' (agent CLI) [default: ollama]",
    )
    ap.add_argument(
        "--model", default=None, metavar="MODEL",
        help="Force a specific model for all sections (overrides auto-selection)",
    )
    ap.add_argument(
        "--paper", default=None, metavar="FILENAME",
        help="Process a single paper by filename (e.g. 'attention.pdf')",
    )
    ap.add_argument(
        "--workers", type=int, default=2, metavar="N",
        help="Parallel paper workers [default: 2 — uses both GPUs]",
    )
    ap.add_argument(
        "--list", action="store_true",
        help="Show processing status for all papers and exit",
    )
    ap.add_argument(
        "--reprocess", default=None,
        metavar="SECTION",
        help="Re-run a specific section for all papers: summary|logic|cpp|diagrams|extras|all",
    )
    ap.add_argument(
        "--override", action="store_true",
        help=(
            "Aggressively provision GPU before processing: evict all loaded "
            "Ollama models (keep_alive=0), and restart the ollama service if "
            "any model is still resident after eviction."
        ),
    )
    ap.add_argument(
        "--verbose", "-v", action="store_true",
        help="Extra debug output",
    )
    args = ap.parse_args()

    papers_dir = Path(args.papers_dir)
    if not papers_dir.exists():
        sys.exit(f"❌  Directory not found: {papers_dir}")

    # ── List mode ──────────────────────────────────────────────────────────
    if args.list:
        list_status(papers_dir)
        return

    _install_signal_handlers()

    # ── Health check ───────────────────────────────────────────────────────
    print(f"\n  🦞  OpenClaw Paper Processor")
    print(f"  {'─'*40}")
    ok = health_check_ollama() if args.backend == "ollama" else health_check_openclaw()
    if not ok:
        sys.exit(1)

    if args.backend == "ollama":
        # Check all models for GPU 0 as a representative baseline
        models_to_check = (
            [args.model]
            if args.model
            else list(dict.fromkeys(
                [GPU_OPTIMIZED_MODELS[0]["reason"], GPU_OPTIMIZED_MODELS[0]["code"], GPU_OPTIMIZED_MODELS[0]["fast"]]
            ))
        )
        check_required_models(models_to_check)

    if args.override and args.backend == "ollama":
        provision_ollama(verbose=args.verbose)

    if args.backend == "ollama":
        prewarm_models()

    # Note: backend URL and models will be dynamically selected per-worker thread
    default_model = args.model or GPU_OPTIMIZED_MODELS[0]["reason"]
    backend       = Backend(args.backend, default_model)

    print(f"  Backend   : {args.backend}")
    print(f"  Default   : {default_model}")
    if args.reprocess:
        print(f"  Reprocess : {args.reprocess}")

    # ── Build file list ────────────────────────────────────────────────────
    if args.paper:
        target = papers_dir / args.paper
        if not target.exists():
            # Fallback: search recursively by basename
            matches = [p for p in papers_dir.rglob(args.paper) if p.is_file()]
            if len(matches) == 1:
                target = matches[0]
            elif len(matches) > 1:
                sys.exit(
                    f"❌  Ambiguous --paper '{args.paper}' — {len(matches)} matches:\n    "
                    + "\n    ".join(str(m.relative_to(papers_dir)) for m in matches[:10])
                )
            else:
                sys.exit(f"❌  File not found: {target}")
        pdfs = [target]
    else:
        pdfs = sorted(
            p for p in papers_dir.rglob("*.pdf") if "_processed" not in p.parts
        )
        if not pdfs:
            sys.exit(f"❌  No PDF files found in {papers_dir}")

    print(f"  Papers    : {len(pdfs)}")
    print(f"  Output    : {papers_dir / '_processed'}")
    print()

    processor = PaperProcessor(
        papers_dir   = papers_dir,
        backend      = backend,
        forced_model = args.model,
        reprocess    = args.reprocess,
        verbose      = args.verbose,
    )

    # ── Process ────────────────────────────────────────────────────────────
    errors: List[str] = []

    def _worker_wrapper(pdf: Path):
        # Quick page-count for GPU routing — fitz.open() on metadata only, very fast
        try:
            _doc = fitz.open(str(pdf))
            _pages = _doc.page_count
            _doc.close()
        except Exception:
            _pages = 0

        gpu_id = None
        with _gpu_lock:
            if not _gpu_available:
                gpu_id = 0  # fallback: all GPUs busy
            elif len(_gpu_available) == 1:
                gpu_id = _gpu_available.pop(0)
            else:
                # Both GPUs free: prefer GPU 1 (RTX 5080, 32k ctx) for long papers
                if _pages > 20 and 1 in _gpu_available:
                    _gpu_available.remove(1)
                    gpu_id = 1
                else:
                    gpu_id = _gpu_available.pop(0)

        _thread_to_gpu[threading.get_ident()] = gpu_id
        print(f"  🚀  Assigned GPU {gpu_id} to {pdf.name} ({_pages} pages)")
        
        try:
            processor.process(pdf)
        finally:
            with _gpu_lock:
                _gpu_available.append(gpu_id)
            _thread_to_gpu.pop(threading.get_ident(), None)

    if args.workers > 1:
        print(f"  ⚡ Parallel mode: {args.workers} workers")
        print(f"  ⚠️   Ensure models fit in VRAM when running concurrently!\n")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker_wrapper, pdf): pdf for pdf in pdfs}
            remaining = set(futures)
            while remaining:
                done, remaining = wait(remaining, timeout=2.0,
                                       return_when=FIRST_COMPLETED)
                for fut in done:
                    pdf = futures[fut]
                    try:
                        fut.result()
                    except _ShutdownRequested:
                        pass
                    except Exception as exc:
                        msg = f"{pdf.name}: {exc}"
                        errors.append(msg)
                        print(f"  ❌  {msg}")
                        if "timed out" in str(exc).lower():
                            print("  🔄  Timeout detected — restarting Ollama to unblock next paper …")
                            _ollama_restart_service()
                if _shutdown.is_set():
                    print(f"  ⚡  Cancelling {len(remaining)} pending paper(s) …")
                    for f in remaining:
                        f.cancel()
                    break
    else:
        for pdf in pdfs:
            if _shutdown.is_set():
                print(f"  ⚡  Shutdown — skipping remaining {len(pdfs) - pdfs.index(pdf)} paper(s)")
                break
            try:
                processor.process(pdf)
            except _ShutdownRequested:
                break
            except Exception as exc:
                msg = f"{pdf.name}: {exc}"
                errors.append(msg)
                print(f"  ❌  {msg}")
                if "timed out" in str(exc).lower():
                    print("  🔄  Timeout detected — restarting Ollama to unblock next paper …")
                    _ollama_restart_service()

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'═'*64}")
    succeeded = len(pdfs) - len(errors)
    print(f"  ✅  {succeeded}/{len(pdfs)} papers processed successfully")
    if errors:
        print(f"  ❌  {len(errors)} error(s):")
        for e in errors:
            print(f"       • {e}")
    print(f"{'═'*64}\n")


if __name__ == "__main__":
    main()
