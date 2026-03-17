# CLAUDE.md — Phase 2: LoRA + Hopfield Reimplementation
# Project: lora-hopfield-clinical
# Sprint: 30-Day AI/ML Portfolio Sprint (Days 12–20)
# Author: Ryan Binny | github.com/RRobin711

---

## How to Read This File

This file is your complete operational context. Before writing any code:

1. Read <project_identity> to understand what this project is and why it exists
2. Read <phase1_context> to understand what was already built and the connecting narrative
3. Read <compute_environment> to understand the hardware setup and ROCm configuration
4. Read <code_standards> — these are non-negotiable and apply to every file you touch
5. Read <phase2_architecture> for the specific implementation plan
6. Read <claudeflow_plan> if you are orchestrating parallel experiment runs

**Before implementing any module:** State your interpretation of what it must do,
what invariants it must preserve, and what the failure modes are. Then implement.
This is not optional — it is the difference between code that works and code that
can be debugged.

---

<project_identity>

## Project Identity

**What this is:** A from-scratch PyTorch reimplementation of two foundational papers:
- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
- Hopfield Networks is All You Need (Ramsauer et al., 2020)

**Why both papers together:** These papers tell a unified story about transformer
attention. Ramsauer et al. proves that transformer attention IS energy minimization
in a modern Hopfield network — it defines *what attention computes*. Hu et al. then
asks: given we understand attention as a fixed-point operation, how do we adapt it
cheaply? LoRA answers by decomposing weight updates into low-rank matrices.
Together: understand the mechanism (Hopfield), exploit it efficiently (LoRA).

**Why this dataset:** DREADDIT (stress detection from Reddit, ~3500 samples, binary
classification). Chosen because:
- Connects to Phase 1 domain: stress is upstream of PHQ-9/GAD-7 clinical scales
- Clean binary metrics make ablation tables interpretable
- Informal Reddit language is a genuine test of associative memory claims in Hopfield
- Fits RX 6700S (8GB VRAM) comfortably with GPT-2 small (20–40 min/run)

**Who will read this code:** ML engineers and hiring managers at applied AI companies.
They will read src/ files. They will check git blame. They will notice if this looks
like generated boilerplate with no engineering judgement applied. Every file must
read like it was written by someone who understood the paper, made deliberate
tradeoffs, and can defend those tradeoffs in an interview.

**The 30-day sprint context:** This is Day 12–20 of a sprint that started with a
deployed Clinical RAG system (Phase 1). The goal is trajectory signaling — showing
that the same engineer who built production retrieval infrastructure also understands
the mathematical foundations of the attention mechanisms underlying the models.

</project_identity>

---

<phase1_context>

## Phase 1 Context: Clinical RAG System (Days 2–11, COMPLETE)

**Live demo:** huggingface.co/spaces/RRobin711/clinical-rag
**Repo:** github.com/RRobin711/Clinical-RAG-system

### What Was Built

A domain-specific RAG system for querying 59 open-access clinical psychology papers
covering psychiatric assessment scales: PHQ-9, MADRS, GAD-7, ADHD-RS.

Architecture:
```
PDF corpus (59 papers)
  -> pypdf ingestion (switched from PyMuPDF due to WSL2 memory issue)
  -> chunking (4 strategies tested: fixed_512, fixed_1024, fixed_2000, semantic)
  -> BAAI/bge-small-en-v1.5 embeddings (384-dim)
  -> FAISS IndexFlatIP (cosine via normalized embeddings)
  -> GPT-4o-mini generation with grounded prompts
```

MLOps: pytest, Docker, GitHub Actions CI, W&B experiment tracking, Gradio demo.

### Key Experimental Results (Phase 1)

Baseline (fixed_512 chunking, naive retrieval):
- Hit@5: 0.850 | Hit@3: 0.825 | MRR: 0.823

HyDE experiment (Gao et al., 2022 — Hypothetical Document Embeddings):
- Expert queries: HyDE *degraded* retrieval in most configs. Best expert config was
  fixed_2000_hyde (Hit@5 = 0.900) — the only configuration where HyDE outperformed
  naive retrieval on expert queries. All smaller chunk sizes saw HyDE hurt performance.
- Layperson queries: HyDE consistently helped (+5-15 Hit@5 points).

**The core finding:** HyDE functions as a vocabulary translation layer. It only adds
value when a genuine terminology gap exists between the query and the corpus. In a
domain-specific corpus where expert queries already share vocabulary with the papers,
HyDE introduces noise rather than bridging a gap. The fixed_2000_hyde config was the
single exception because large chunks provide enough surrounding context that the
noise is amortized.

### Why This Is Relevant to Phase 2

Phase 1 identified that the retrieval bottleneck is semantic matching — getting the
right embedding for the right reason. Phase 2 implements the mechanisms underneath
that problem:

- **LoRA** is how you would fine-tune the bge embedding model on clinical text without
  full fine-tuning cost (Hu et al. show r=4 recovers most full fine-tune quality;
  r=8 is a conservative choice — Phase 2 ablation will verify this on DREADDIT)
- **Hopfield attention** provides a theoretical account of why larger chunks helped
  HyDE: they create "denser" memory patterns with more associative overlap, lowering
  the energy barrier to retrieval
- The DREADDIT dataset (informal Reddit stress text) mirrors the layperson query
  failure mode in Phase 1 — both involve informal language hitting a specialized corpus

This narrative must appear in:
- The Phase 2 README
- Docstrings in src/model.py and src/hopfield.py
- The blog post
Claude Code: when writing docstrings and comments, use this framing. Do not write
generic "this implements LoRA" comments — write comments that explain *why* the
design choice matters given the Phase 1 context above.

### Engineering Lessons Carried Forward from Phase 1

1. **WSL2 memory management:** C-extension libraries (PyMuPDF) caused unrecoverable
   memory accumulation. Pure-Python equivalents were GC-safe. In Phase 2: prefer
   pure-PyTorch operations over mixed C-extension calls in hot loops.

2. **Evaluation before optimization:** Phase 1 built the gold QA dataset before
   running any experiments. Phase 2 must do the same: define the comparison table
   schema (frozen baseline / LoRA rank-4 / LoRA rank-16 / Hopfield / full fine-tune)
   before writing a single training loop.

3. **Reproducibility is non-negotiable:** Every experiment result in Phase 1 was
   logged to W&B with full config. Phase 2 uses the same discipline:
   torch.manual_seed, numpy seed, dataloader worker seed, logged as run config.

4. **Negative results are the point:** The HyDE finding was the most valuable output
   of Phase 1 precisely because it was negative and explainable. Phase 2 should
   be designed to produce results that can be explained, not just results that look good.

</phase1_context>

---

<compute_environment>

## Compute Environment

### Hardware

**Machine:** Laptop with AMD Radeon RX 6700S GPU (8GB VRAM, RDNA 2, gfx1032)
**OS for development:** Ubuntu 24.04 LTS installed on external HDD via USB 3.0
**Why external drive:** Internal drive stays untouched (Windows + games/files).
Ubuntu boots from the external drive. Once Python, model, and data are in RAM/VRAM,
the HDD is not a bottleneck — training is GPU-bound.

**Performance notes (HDD-specific):**
- Boot: ~45-60 seconds (slower than SSD, acceptable)
- Package installs: slower writes, budget extra 15-20 min for initial setup
- Training: full GPU speed once data is loaded into memory (DREADDIT is ~3500 samples,
  fits in RAM trivially; GPT-2 small is ~500MB, fits in 8GB VRAM easily)
- Recommendation: avoid unnecessary large file I/O during training. Load dataset
  once, keep in memory. Do not re-read from disk each epoch.

### ROCm Setup

The RX 6700S (gfx1032) is not on the official ROCm support list but works on
native Linux by masquerading as gfx1030 (which IS officially supported, same RDNA 2
architecture family). This is a well-documented community workaround.

**Required environment variable (set in ~/.bashrc):**
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

This must be set BEFORE any PyTorch import. Without it, `torch.cuda.is_available()`
returns False. With it, the GPU works normally through the standard `torch.cuda` API.

**NOTE:** This override does NOT work on WSL2 (RDNA 2 lacks WSL2 ROCm support due
to PCIe atomics limitations in the virtualization layer). This is why we use native
Linux on the external drive, not WSL2.

**PyTorch install (ROCm build):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

**Verification (run after install):**
```python
import torch
print(torch.cuda.is_available())       # Should be True
print(torch.cuda.get_device_name(0))   # Should show "AMD Radeon RX 6700S"
t = torch.randn(100, 100).cuda()       # Should work without hanging
print(t.sum())                          # Should print a number
```

If `torch.cuda.is_available()` returns False, check:
1. `HSA_OVERRIDE_GFX_VERSION=10.3.0` is exported
2. ROCm is installed (`rocminfo` should list gfx1032 as an agent)
3. User is in `video` and `render` groups (`sudo usermod -a -G video,render $USER`)

### Environment Details

**Python:** 3.11+ via system package or pyenv
**Venv:** ~/projects/lora-hopfield-clinical/.venv
**Key packages:** torch (ROCm build), transformers, datasets, wandb, scikit-learn,
  pytest, tqdm
**Node:** install via nvm for Claude Code
**Claude Code:** install per Anthropic docs after nvm/node setup
**claude-flow:** npm install -g claude-flow (used for parallel ablation runs)

**W&B project name:** lora-hopfield-clinical

### Code Portability Rule

All `src/` modules must work on both CPU and CUDA (ROCm presents as CUDA to PyTorch).
Never hardcode `"cuda"` — always use:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
Tests run on CPU (fast, no GPU dependency). Training runs on GPU. Same code handles both.

</compute_environment>

---

<code_standards>

## Code Standards (Non-Negotiable)

These apply to every file. Violations will make the codebase look AI-generated.
Hiring managers who read PyTorch code can distinguish "LLM wrote this" from
"engineer who understands PyTorch wrote this" in about 30 seconds.

### Typing and Imports

REQUIRED:
```python
from __future__ import annotations
# Python 3.11: prefer lowercase built-in generics (dict, list, tuple)
# and X | None over Optional[X]. Old-style typing imports still work but
# are considered legacy style in 3.10+.
import torch
import torch.nn as nn
import torch.nn.functional as F
```

Every function must have complete type annotations on all parameters and return types.
No `Any` unless genuinely unavoidable (and comment explaining why).

WRONG — do not do this:
```python
def forward(self, x):
    return self.linear(x)
```

RIGHT (Python 3.11 style):
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear(x)

def inject_lora(model: nn.Module, config: LoRAConfig) -> dict[str, int]:
    # dict[str, int] not Dict[str, int] — Python 3.11 style
    ...
```

### Docstrings

Every class and every public method needs a docstring. Docstrings must include:
- What the class/function does (1-2 lines)
- The mathematical operation it implements, if any (use LaTeX notation in comments)
- Args with types and shapes for tensor operations
- Returns with shape
- References to the paper section/equation number

WRONG:
```python
class LoRALinear(nn.Module):
    """LoRA linear layer."""
    def forward(self, x):
        ...
```

RIGHT:
```python
class LoRALinear(nn.Module):
    """
    Low-rank adaptation of a frozen nn.Linear layer.

    Implements the LoRA reparameterization from Hu et al. (2021), Section 4:
        h = W_0 x + (B A) x * (alpha / r)
    where W_0 is frozen, A is initialized with Kaiming uniform, B is zero-initialized
    so the adapter contributes nothing at the start of training.

    This zero-init of B is critical: it means the model starts from exactly the
    pretrained checkpoint, not a random perturbation of it.

    Args:
        in_features: Input dimension d_in
        out_features: Output dimension d_out
        r: LoRA rank. Paper ablates r in {1, 2, 4, 8, 16, 64}.
           r=4 recovers most full fine-tune performance; higher ranks yield
           diminishing returns while increasing parameter count.
        alpha: Scaling factor. Effective lr for adapter = lr * alpha/r.
               Setting alpha=r makes the scale invariant to r choice.
        dropout: Applied to input before the low-rank projection.

    Example:
        >>> layer = LoRALinear(768, 768, r=4, alpha=4)
        >>> x = torch.randn(2, 10, 768)
        >>> out = layer(x)  # shape: (2, 10, 768)
    """
```

### PyTorch Patterns

Use `nn.Module` for everything. No raw function-style forward passes for anything
that has learnable parameters. Use `register_buffer` for non-trainable tensors that
should move with `.to(device)`. Use `nn.Parameter` for learnable tensors not managed
by submodules.

WRONG — tensor created in forward (recreated every call, not moved by .to()):
```python
def forward(self, x):
    beta = torch.ones(self.n) * self.beta_val
    return softmax(beta * x @ x.T) @ x
```

RIGHT:
```python
def __init__(self, n: int, beta: float):
    super().__init__()
    self.register_buffer("beta", torch.tensor(beta))
```

Gradient checks: any new `nn.Module` must have a corresponding test that verifies
gradients flow to trainable parameters and do NOT flow to frozen parameters.

```python
def test_lora_gradients():
    layer = LoRALinear(64, 64, r=4, alpha=4)
    x = torch.randn(2, 64, requires_grad=True)
    loss = layer(x).sum()
    loss.backward()
    # A and B should have gradients
    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None
    # Frozen base weight should NOT
    assert layer.weight.grad is None
```

### Numerical Stability

Any operation involving softmax over large sequences must use the log-sum-exp trick
or pass through `F.softmax(..., dim=-1)`. Never implement raw exp/sum manually.

For Hopfield retrieval: the energy function involves exp(beta * similarity). With
large beta or high-dimensional embeddings, this overflows. Use:
```python
# Stable: subtract max before exp
# NOTE: use .transpose(-2, -1), NOT .T
# .T on a 3D tensor (B, M, d) transposes ALL dims -> (d, M, B), which is wrong.
# .transpose(-2, -1) only swaps the last two dims -> (B, d, M), which is correct.
scores = beta * (queries @ keys.transpose(-2, -1))  # (B, N, d) @ (B, d, M) -> (B, N, M)
attn = F.softmax(scores, dim=-1)    # numerically stable
```
Document why stability matters in the Hopfield docstring — this is a known failure
mode Ramsauer et al. address explicitly in their paper (check the appendix of your
downloaded version for the stability analysis).

### Configuration

All hyperparameters go in a dataclass, never as magic numbers scattered in code.

```python
@dataclass
class LoRAConfig:
    r: int = 4
    alpha: float = 4.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("c_attn",)  # GPT-2 attention projection name
```

This makes W&B logging trivial (`wandb.config.update(asdict(cfg))`) and makes
ablations a one-line parameter change.

### Error Messages

Raise with context, not bare asserts.

WRONG:
```python
assert r > 0
```

RIGHT:
```python
if r <= 0:
    raise ValueError(
        f"LoRA rank r must be positive, got r={r}. "
        "Common values: r=4 (efficient), r=16 (near full fine-tune quality)."
    )
```

### File Length and Cohesion

Each file has one job. If a file exceeds ~300 lines, it likely has two jobs.
Split it. The structure is:
- `src/lora.py` — LoRALinear only
- `src/hopfield.py` — HopfieldAttention only
- `src/model.py` — GPT-2 injection (imports from lora.py and hopfield.py)
- `src/train.py` — training loop (imports from model.py)
- `src/evaluate.py` — evaluation metrics (no model imports — takes predictions)

Circular imports are a sign of wrong abstraction. If you see a circular import,
stop and redesign before writing more code.

### Commits

Every commit message must be:
```
<type>(<scope>): <what changed and why>

[optional body: what tradeoff was made or what was learned]
```

Types: feat, fix, refactor, test, docs, chore
Examples:
- `feat(lora): implement LoRALinear with zero-init B per Hu et al. Section 4`
- `test(lora): add gradient flow test verifying frozen base weight`
- `fix(hopfield): clip beta to prevent softmax overflow on long sequences`
- `refactor(train): extract seed_everything() to remove duplication`

BAD commit messages that signal LLM-generated code:
- "Add files"
- "Update implementation"
- "Fix bug"
- "Implement the requested changes"

</code_standards>

---

<phase2_architecture>

## Phase 2 Architecture

### Directory Structure

```
lora-hopfield-clinical/
├── src/
│   ├── lora.py            # LoRALinear module — wraps nn.Linear
│   ├── hopfield.py        # HopfieldAttention module — energy-minimizing attention
│   ├── model.py           # GPT-2 + injection logic for LoRA and Hopfield
│   ├── train.py           # Shared training loop (works for all model configs)
│   ├── evaluate.py        # Accuracy, F1, confusion matrix (no model deps)
│   └── data.py            # DREADDIT loading, tokenization, DataLoader factory
├── scripts/
│   ├── run_lora_ablation.py    # Rank ablation: r in {1, 4, 8, 16, 32}
│   ├── run_comparison.py       # 4-way comparison table
│   └── aggregate_results.py    # Combine ablation JSON files into summary table
├── tests/
│   ├── test_lora.py
│   ├── test_hopfield.py
│   ├── test_model.py
│   └── test_evaluate.py
├── notebooks/
│   └── visualize_results.ipynb  # Visualization only — NOT main interface
├── results/                      # JSON result files, W&B artifacts
│   └── checkpoints/              # Training checkpoints (gitignored)
├── CLAUDE.md
└── README.md
```

### Implementation Order (Days 12–20)

**Days 12-13: LoRALinear**
- Implement `LoRALinear` in `src/lora.py`
- Key: B initialized to zero, A initialized Kaiming uniform
- Key: scale = alpha / r applied at forward time (not baked into weights)
- Key: base weight `nn.Linear.weight` must be frozen (`requires_grad=False`)
- Write `tests/test_lora.py` before finishing Day 12:
  - gradient flow test (above)
  - shape preservation test
  - zero-init test: `assert torch.allclose(layer.lora_B, torch.zeros_like(layer.lora_B))`
  - parameter count test: verify `sum(p.numel() for p in model.parameters() if p.requires_grad)`
    equals `r * (in_features + out_features)` (plus bias if used).
    Breakdown: A is shape (r, in_features) -> r*in params; B is shape (out_features, r) -> r*out params.
    Total = r*(in+out). A common off-by-one error is doubling this — don't.

**Day 14: Data + Training Infrastructure**
- `src/data.py`: DREADDIT from HuggingFace (`dkmkknub/dreaddit`), GPT-2 tokenizer,
  max_length=256, return_tensors="pt", attention_mask handled correctly.
  **HDD note:** Load dataset into memory once at startup. Use `dataset.with_format("torch")`
  or cache tokenized data to avoid re-reading from disk each epoch.
- `src/train.py`: training loop with:
  - `seed_everything(seed: int)` — sets torch, numpy, random, cuda seeds
  - W&B run init with full config logged
  - gradient clipping (`torch.nn.utils.clip_grad_norm_`, max_norm=1.0)
  - learning rate scheduler (linear warmup + cosine decay)
  - checkpoint saving to `results/checkpoints/`
  - early stopping on validation loss (patience=3)
  - device-agnostic: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- `src/evaluate.py`: takes `y_true, y_pred` numpy arrays, returns dict with
  accuracy, F1 (macro), confusion matrix — no model dependency
- `scripts/aggregate_results.py`: reads `results/ablation_r*.json` or
  `results/comparison_*.json` files via glob pattern, prints sorted comparison table.
  **Write this before running any ablation** so results are immediately consumable.

**Days 15-16: LoRA Training + Rank Ablation**
- `src/model.py`: GPT-2 loading + LoRA injection
  - Load `gpt2` from HuggingFace, freeze all base weights
  - **Critical GPT-2 gotcha:** HuggingFace GPT-2 uses `Conv1D` (from `transformers.pytorch_utils`),
    NOT `nn.Linear`. Conv1D stores weights transposed: shape is (in_features, out_features)
    vs nn.Linear's (out_features, in_features). Your LoRALinear wraps nn.Linear —
    you must either (a) replace Conv1D layers by converting the weight on load, or
    (b) write a LoRAConv1D variant. Option (a) is simpler; document the transposition.
  - Replace `c_attn` (combined QKV projection, shape 768->2304 for gpt2-small) in each
    attention block with your LoRA-wrapped layer
  - Add classification head (linear layer on last non-padding token hidden state)
  - `inject_lora(model, config: LoRAConfig)` must be a standalone function, not
    baked into __init__
  - Print parameter table: total params, trainable params, trainable %
- `scripts/run_lora_ablation.py`: runs ranks [1, 4, 8, 16, 32] sequentially,
  saves results JSON per run, prints final comparison table.
  Can also be run per-rank: `python scripts/run_lora_ablation.py --rank 4 --seed 42 --output results/ablation_r4.json`
  This single-rank mode is what claude-flow sub-agents invoke.

**Day 17: Hopfield Attention**
- `src/hopfield.py`: `HopfieldAttention` as a module that replaces the attention
  computation inside GPT-2's `GPT2Attention` class.
  - NOTE: GPT-2 does NOT use `nn.MultiheadAttention`. It uses a bespoke `GPT2Attention`
    class (in transformers/models/gpt2/modeling_gpt2.py) with Conv1D projections and
    manual QKV splitting. You are replacing the core attention weight computation
    (the softmax(QK^T/sqrt(d)) @ V step), not swapping out an nn.MultiheadAttention.
  - Implement the continuous Hopfield update rule from Ramsauer et al. (2020) —
    the section titled "Modern Hopfield Networks" in your downloaded PDF: the retrieval
    step is a fixed-point iteration that converges to softmax attention as a special case.
    (Do not hardcode section numbers — verify against your PDF before coding.)
  - Beta parameter: controls pattern separation. High beta -> winner-take-all retrieval
    (sharp attention). Low beta -> distributed retrieval (soft attention). This is the
    inverse temperature from the energy function.
  - `update_rule(xi, stored_patterns, beta)` as a pure function for testability
  - Test: verify output shape matches standard attention output, verify that at
    beta ~ 1/sqrt(d_head) the output approximates scaled dot-product attention

**Day 18: Comparison Table**
- `scripts/run_comparison.py`: 4 configurations on DREADDIT test set:
  1. Frozen GPT-2 (no fine-tuning) — lower bound
  2. LoRA r=4 (best from ablation)
  3. HopfieldAttention replacement
  4. Full fine-tuning — upper bound
- Log all 4 to same W&B project (`lora-hopfield-clinical`) for clean comparison charts
- Output: `results/comparison_table.json` with accuracy, F1, trainable params,
  training time per epoch

**Days 19-20: Blog Post + Cleanup**
- Blog title: "Reimplementing LoRA and Hopfield Attention: What the Papers Don't Tell You"
- Required sections:
  1. The unified narrative (what Hopfield tells us about attention; why LoRA follows)
  2. The implementation gap (what papers omit: zero-init, alpha/r scaling, seed discipline)
  3. Rank ablation results (chart + interpretation — where does the curve flatten?)
  4. Connection back to Phase 1 (how LoRA would change the RAG retrieval story)
  5. What failed (numerical stability in Hopfield, training instability at r=1)

</phase2_architecture>

---

<claudeflow_plan>

## claude-flow Orchestration Plan

**When to use claude-flow:** Days 15-16, after `src/lora.py`, `src/model.py`,
and `src/train.py` are verified working (all tests pass, single training run completes).
Do NOT use claude-flow before the single-agent implementation is confirmed correct.
Parallel runs on broken code produce parallel garbage.

**What to parallelize:** The rank ablation (r in {1, 4, 8, 16, 32}) and the 4-way
comparison runs are embarrassingly parallel — each run is independent and writes
to a separate results file.

**GPU contention note:** The RX 6700S has 8GB VRAM. GPT-2 small + LoRA uses ~2-3GB.
Running 2 parallel training jobs may fit in VRAM; running 5 simultaneously will NOT.
Use claude-flow with concurrency=1 for GPU training runs. The value is automated
sequential orchestration — each agent starts the next rank after the previous one
finishes, writes results, and the orchestrator aggregates at the end. This saves
you from babysitting 5 sequential runs manually. For truly parallel work, use
claude-flow for non-GPU tasks (test writing, docs, result aggregation) while a
training run is active on GPU.

### Orchestrator Prompt Template

When initiating the ablation swarm, the orchestrator should be prompted with:

```
You are coordinating a LoRA rank ablation study across 5 sequential agents.
Each agent runs exactly one configuration and writes results to a unique JSON file.
Do not let agents share state or write to the same file.
Run agents ONE AT A TIME (GPU memory constraint: 8GB VRAM, only one training
job fits comfortably).

<task>
Run the LoRA rank ablation for the lora-hopfield-clinical project.
Each sub-agent receives one rank value and must:
1. Export HSA_OVERRIDE_GFX_VERSION=10.3.0
2. Run: python scripts/run_lora_ablation.py --rank {r} --seed 42 --output results/ablation_r{r}.json
3. Verify the output file was written and contains keys: rank, accuracy, f1, trainable_params, epochs
4. Report back: rank value, final val accuracy, any errors
Do not modify any source files. Do not run more than the assigned rank.
</task>

<agents>
Agent 1: rank=1
Agent 2: rank=4
Agent 3: rank=8
Agent 4: rank=16
Agent 5: rank=32
</agents>

After all agents complete, aggregate results from results/ablation_r*.json
and print a sorted comparison table.
```

### Sub-Agent Constraints

Each sub-agent must:
- Read CLAUDE.md before any action (to pick up code standards)
- Export `HSA_OVERRIDE_GFX_VERSION=10.3.0` before running any Python
- Write only to `results/ablation_r{rank}.json` — no other file writes
- Not modify source files
- Exit cleanly on error (write error state to results file, do not crash silently)

### Result Aggregation

After swarm completes, run locally:
```bash
python scripts/aggregate_results.py --pattern "results/ablation_r*.json"
```

This script must be written before initiating the swarm so results are
immediately consumable.

### What claude-flow Is NOT For

- Writing source code (single-agent, careful implementation only)
- Running tests (run pytest locally before any swarm)
- Git operations (all commits are manual — commit messages must reflect your
  understanding, not an agent's summary)

</claudeflow_plan>

---

<session_startup_checklist>

## Session Startup Checklist

When Claude Code opens this project, before writing any code:

1. Verify `HSA_OVERRIDE_GFX_VERSION=10.3.0` is set (echo $HSA_OVERRIDE_GFX_VERSION)
2. Run `pytest tests/ -v` — all tests must pass before new code is written
3. Run `git status` — no uncommitted changes should exist from a prior session
4. State which Day (12-20) is being worked on and what the end-of-day deliverable is
5. If implementing a new module: state the paper section and equation number being
   implemented before writing any code

If tests are failing at session start, fix them before proceeding. Broken tests
carried forward produce compounding technical debt.

</session_startup_checklist>

---

<do_not_do>

## Explicit Prohibitions

These patterns make the codebase look LLM-generated. Do not use them.

- Do not use `# TODO: implement this` placeholders in committed code
- Do not write `pass` as a function body in committed code
- Do not name variables `result`, `output`, `data`, `temp` without qualification
- Do not write comments that restate what the code does (`# multiply A and B`)
  — write comments that explain *why* (`# zero-init B so adapter = identity at t=0`)
- Do not implement features not in the architecture plan without flagging them
- Do not run git add . — stage files individually and verify each before committing
- Do not write training loops without explicit seed setting
- Do not write a new module without a corresponding test file
- Do not hardcode `"cuda"` — always use a device variable so code runs on CPU (tests)
  and GPU (training) without changes

</do_not_do>

---

<sprint_progress>

## Sprint Progress (Updated 2026-03-16)

### Completed Days

**Day 12: src/lora.py — COMPLETE**
- LoRALinear with zero-init B, Kaiming-uniform A, frozen base weight
- LoRAConfig dataclass, from_linear() classmethod for Conv1D conversion path
- 31/31 tests passing (test_lora.py)

**Day 13: src/hopfield.py — COMPLETE**
- HopfieldAttention with configurable β (inverse temperature)
- hopfield_retrieval() pure function for testability
- At β = 1/√d_head, exactly recovers standard scaled dot-product attention
- Multi-iteration support (converges toward Hopfield energy minimum)
- 30/30 tests passing (test_hopfield.py)

**Day 14: src/data.py, src/train.py, src/evaluate.py — COMPLETE**
- data.py: DREADDIT loading, GPT-2 tokenization, stratified train/val/test split
- train.py: shared training loop with W&B, early stopping, LR scheduling,
  optimizer on trainable params only (filter requires_grad)
- evaluate.py: accuracy, F1 macro, confusion matrix, JSON-serializable output
- 33/33 tests passing (test_data.py, test_train.py, test_evaluate.py)

**Day 15: src/model.py — COMPLETE**
- GPT-2 loading via GPT2Model (NOT GPT2LMHeadModel — train.py needs last_hidden_state)
- conv1d_to_linear(): weight transpose (in,out) -> (out,in) for Conv1D -> nn.Linear
- inject_lora(): standalone function, freezes all base weights, replaces target Conv1D
- verify_lora_injection(): post-injection diagnostic (param counts, freeze checks)
- freeze_all_parameters() / unfreeze_all_parameters() for baseline + full fine-tune
- 26/26 tests passing (test_model.py), smoke test passed on CPU

**Total test suite: 120/120 passing**

### Key Facts for Day 16

- DREADDIT dataset path: `andreagasparini/dreaddit` (NOT `dkmkknub/dreaddit` — that path is dead)
- Model class: `GPT2Model` (base transformer) — `outputs.last_hidden_state` exists here,
  not on `GPT2LMHeadModel` which returns `.logits` instead
- Conv1D -> Linear: weight must be transposed. Conv1D stores (in, out), nn.Linear stores (out, in).
  Verified numerically equivalent with test_model.py::TestConv1dToLinear
- LoRA r=4 on c_attn across 12 blocks: 147,456 trainable adapter params (0.12% of 124M total)
- Smoke test ran on CPU only — GPU verification (ROCm/RX 6700S) is Day 16 first action
- Classification head: nn.Linear(768, 2) created in train.py, extracts last non-pad token
  hidden state via attention_mask.sum(-1) - 1
- W&B project: `lora-hopfield-clinical`
- Run naming convention: `"lora-r{rank}-s{seed}"`, `"frozen-baseline-s{seed}"`, `"full-finetune-s{seed}"`

### Day 16 Deliverables

1. GPU smoke test (HSA_OVERRIDE_GFX_VERSION=10.3.0, verify CUDA available)
2. scripts/run_lora_ablation.py — single-rank and sequential-all modes
3. scripts/aggregate_results.py — must exist BEFORE running any ablation
4. First full training run: LoRA r=4, seed=42, on GPU
5. Rank ablation: r in {1, 4, 8, 16, 32}

</sprint_progress>
