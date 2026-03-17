# Technical Extraction — lora-hopfield-clinical
# Source material for narrative report. Facts and numbers only.
# Generated: 2026-03-16

---

## 1. Test Suite Summary

**Platform:** Python 3.12.3, pytest 9.0.2, pluggy 1.6.0
**Result:** 137 passed in 28.52s. Zero failures.

| Test File | Tests | Pass | Fail |
|---|---|---|---|
| tests/test_data.py | 9 | 9 | 0 |
| tests/test_evaluate.py | 10 | 10 | 0 |
| tests/test_hopfield.py | 30 | 30 | 0 |
| tests/test_hopfield_gpt2.py | 17 | 17 | 0 |
| tests/test_lora.py | 31 | 31 | 0 |
| tests/test_model.py | 26 | 26 | 0 |
| tests/test_train.py | 14 | 14 | 0 |
| **Total** | **137** | **137** | **0** |

**Note:** CLAUDE.md sprint_progress section states 120/120; actual suite is 137/137 (17 additional tests from test_hopfield_gpt2.py added Day 17).

### Test Classes That Reveal Design Decisions

| Test Class | File | What It Validates |
|---|---|---|
| TestBetaRecovery | test_hopfield_gpt2.py | At β=1/√d_head, Hopfield-injected GPT-2 produces IDENTICAL output to vanilla GPT-2 (±1e-4) |
| TestStandardAttentionRecovery | test_hopfield.py | Hopfield retrieval at default β exactly recovers scaled dot-product attention (±1e-5) |
| TestConv1dToLinear | test_model.py | Conv1D (in,out) → nn.Linear (out,in) transposition is numerically equivalent |
| TestOutputEquivalenceAtInit | test_model.py | LoRA model ≡ vanilla GPT-2 at initialization (zero-init B invariant) |
| TestNumericalStability | test_hopfield.py | No NaN/inf at extreme β values or large input magnitudes |
| TestMultiIteration | test_hopfield.py | Multi-iteration Hopfield retrieval converges toward fixed point |
| TestOptimizerFiltering | test_train.py | Optimizer only sees trainable params; frozen params unchanged after step |
| TestGradientFlow | test_lora.py, test_hopfield.py, test_hopfield_gpt2.py | Gradients flow to adapter/QKV params, NOT to frozen base weights |

---

## 2. Result Files — Unified Table

### Primary Ablation (early stopping, patience=3, max 30 epochs)

| Config | Rank | Trainable Params | Trainable % | Test Acc | Test F1 (macro) | Best Epoch | Best Val Loss | Training Time (s) | W&B Run ID |
|---|---|---|---|---|---|---|---|---|---|
| Frozen baseline | 0 | 1,538 | 0.0012% | 0.6741 | 0.6730 | 30 | 0.5559 | 1,762.8 | yq5x52zv |
| LoRA r=1 | 1 | 38,402 | 0.0309% | 0.6951 | 0.6951 | 30 | 0.5238 | 3,118.6 | vuykn55l |
| LoRA r=4 | 4 | 148,994 | 0.1196% | 0.7259 | 0.7254 | 26 | 0.4904 | 2,976.7 | 348htjeh |
| LoRA r=8 | 8 | 296,450 | 0.2377% | 0.7427 | 0.7426 | 18 | 0.4628 | 2,208.1 | 3qh60b02 |
| LoRA r=16 | 16 | 591,362 | 0.4730% | 0.7538 | 0.7538 | 18 | 0.4393 | 2,179.2 | g8mkyemx |
| LoRA r=32 | 32 | 1,181,186 | 0.9403% | 0.7664 | 0.7655 | 17 | 0.4171 | 2,100.2 | pq2plsoj |
| LoRA r=64 | 64 | 2,360,834 | 1.8618% | 0.7650 | 0.7647 | 11 | 0.4138 | 1,470.2 | 0pob6ww7 |
| LoRA r=128 | 128 | 4,720,130 | 3.6545% | 0.7692 | 0.7670 | 13 | 0.4136 | 1,678.5 | ggr67mmv |

### Epoch-10-Capped Ablation (max 10 epochs, no early stopping trigger)

| Config | Rank | Trainable Params | Test Acc | Test F1 (macro) | Best Epoch | Best Val Loss | Training Time (s) | W&B Run ID |
|---|---|---|---|---|---|---|---|---|
| Frozen baseline | 0 | 1,538 | 0.5860 | 0.5804 | 10 | 0.6536 | 581.9 | 97u6rrst |
| LoRA r=1 | 1 | 38,402 | 0.5916 | 0.5843 | 10 | 0.6464 | 1,030.5 | tjzfzj67 |
| LoRA r=4 | 4 | 148,994 | 0.6056 | 0.6017 | 10 | 0.6264 | 1,047.6 | mz14fl4y |
| LoRA r=8 | 8 | 296,450 | 0.6448 | 0.6442 | 10 | 0.6029 | 1,049.8 | dugh977c |
| LoRA r=16 | 16 | 591,362 | 0.6643 | 0.6643 | 10 | 0.5715 | 1,048.2 | ivaw4khc |
| LoRA r=32 | 32 | 1,181,186 | 0.6895 | 0.6895 | 10 | 0.5361 | 1,053.1 | 4b91vkz9 |
| LoRA r=64 | 64 | 2,360,834 | 0.7189 | 0.7187 | 10 | 0.4878 | 1,053.0 | wcq4d1i7 |
| LoRA r=128 | 128 | 4,720,130 | 0.7538 | 0.7524 | 10 | 0.4679 | 1,073.9 | dq99fwc0 |

### Hopfield Attention

| Field | Value |
|---|---|
| Config | hopfield |
| Seed | 42 |
| Beta | 0.125 (= 1/√64 = 1/√d_head) |
| Num iterations | 1 |
| Trainable params | 1,538 (classification head only) |
| Total params | 124,441,346 |
| Trainable % | 0.0012% |
| Test accuracy | 0.6811 |
| Test F1 macro | 0.6772 |
| Best val loss | 0.5496 |
| Best epoch | 30 |
| Training time | 1,724.5s |
| W&B run ID | 3fx5qwm1 |

### Full Fine-Tune (Bad Run — lr=2e-5, too aggressive)

| Field | Value |
|---|---|
| Config | full |
| Seed | 42 |
| LR | 2e-5 (default, inherited from LoRA config) |
| Trainable params | 124,441,346 (100%) |
| Test accuracy | 0.7566 |
| Test F1 macro | 0.7504 |
| Best val loss | 0.4196 |
| Best epoch | 2 |
| Training time | 661.4s |
| W&B run ID | 7ex5isv2 |

### Full Fine-Tune (Canonical Run — lr=5e-6)

| Field | Value |
|---|---|
| Config | full |
| Seed | 42 |
| LR | 5e-6 |
| Trainable params | 124,441,346 (100%) |
| Test accuracy | 0.7664 |
| Test F1 macro | 0.7640 |
| Best val loss | 0.3988 |
| Best epoch | 5 |
| Training time | 1,119.6s |
| W&B run ID | 5mocwj86 |

### ablation_summary.csv (matches primary ablation table, ranks 0–128)

Confirmed: CSV contains 8 rows matching the primary ablation results above exactly.

---

## 3. Source File Audit

### src/lora.py

**Classes:**
- `LoRAConfig` (dataclass): `r: int = 4`, `alpha: float = 4.0`, `dropout: float = 0.0`, `target_modules: tuple[str, ...] = ("c_attn",)`
- `LoRALinear(nn.Module)`: `__init__(self, in_features: int, out_features: int, r: int = 4, alpha: float = 4.0, dropout: float = 0.0, bias: bool = True) -> None`

**Alpha = rank design choice:**
- `self.scaling = alpha / r` — makes effective LR invariant to rank during ablation. With alpha=r (the default), scaling=1.0 regardless of rank.

**Key design decisions:**
- B zero-initialized (`torch.zeros`), A Kaiming-uniform (`nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))`)
- Base weight frozen via `requires_grad = False` on weight and bias
- `lora_A` shape: `(r, in_features)`, `lora_B` shape: `(out_features, r)`
- Forward: `base_out + F.linear(F.linear(dropout(x), lora_A), lora_B) * scaling`
- `from_linear()` classmethod: copies weights from existing nn.Linear, creates new LoRALinear with independent memory

### src/hopfield.py

**Classes:**
- `HopfieldConfig` (dataclass): `beta: float | None = None`, `num_iters: int = 1`, `dropout: float = 0.0`
- `HopfieldAttention(nn.Module)`: `__init__(self, d_head: int, config: HopfieldConfig | None = None) -> None`

**Functions:**
- `hopfield_retrieval(queries, keys, values, beta, causal_mask, dropout_p, training) -> torch.Tensor` — pure function, no learnable parameters

**Key design decisions:**
- Default beta: `1.0 / (d_head ** 0.5)` — recovers standard scaled dot-product attention
- `register_buffer("beta", torch.tensor(beta_val, dtype=torch.float32))` — non-trainable, moves with `.to(device)`
- Numerical stability: `F.softmax(scores, dim=-1)` handles log-sum-exp internally
- Causal mask: `scores.masked_fill(causal_mask, float("-inf"))` before softmax
- Multi-iteration: queries updated iteratively (`result = hopfield_retrieval(result, keys, values, ...)`)
- Uses `.transpose(-2, -1)` not `.T` (documented: `.T` on 3D tensor transposes ALL dims)

### src/model.py

**Functions:**
- `conv1d_to_linear(conv1d: Conv1D) -> nn.Linear`
  - `linear.weight.data = conv1d.weight.data.T.contiguous()` — transpose (in,out) → (out,in)
  - `linear.bias.data = conv1d.bias.data.clone()`
- `load_gpt2(model_name: str = "gpt2") -> GPT2Model` — returns base model (NOT GPT2LMHeadModel)
- `freeze_all_parameters(model: nn.Module) -> None`
- `unfreeze_all_parameters(model: nn.Module) -> None`
- `inject_lora(model: GPT2Model, config: LoRAConfig) -> dict[str, int]`
  - Freezes all params first
  - For each of 12 blocks: get Conv1D → convert to nn.Linear → wrap with LoRALinear.from_linear() → setattr
  - Returns dict mapping module paths to trainable param counts
- `verify_lora_injection(model: GPT2Model, config: LoRAConfig) -> dict[str, object]`
  - Checks: trainable count = r*(in+out), base frozen, adapters trainable
- `print_param_table(model: nn.Module, head: nn.Module | None = None) -> None`

### src/hopfield_gpt2.py

**Functions:**
- `_make_hopfield_forward(original_attn: nn.Module, hopfield: HopfieldAttention) -> callable`
  - Replaces the attention computation inside GPT2Attention
  - Preserves: c_attn (QKV projection), c_proj (output projection), resid_dropout
  - Replaces: the softmax(QK^T/√d)V step with Hopfield retrieval
  - **Mask convention inversion:** GPT-2 uses True=attend; Hopfield uses True=masked. Conversion: `causal_mask = ~attention_mask` (boolean) or `causal_mask = attention_mask < -1.0` (float additive)
  - When no mask provided: constructs standard causal mask via `torch.triu(..., diagonal=1)`
  - Attaches HopfieldAttention as `attn.hopfield_attn` (proper submodule for .to(device))
- `inject_hopfield(model: GPT2Model, config: HopfieldConfig | None = None) -> dict[str, int]`
  - Freezes all params
  - Computes `d_head = hidden_size // num_attention_heads` (768//12 = 64)
  - Creates HopfieldAttention per block, attaches, replaces forward via `types.MethodType`
  - Returns dict mapping paths to 0 (no trainable params added)
- `verify_hopfield_injection(model: GPT2Model) -> dict[str, object]`

### src/data.py

**Classes:**
- `DataConfig` (dataclass): `dataset_name: str = "andreagasparini/dreaddit"`, `max_length: int = 256`, `batch_size: int = 16`, `val_fraction: float = 0.2`, `seed: int = 42`, `num_workers: int = 0`

**Functions:**
- `tokenize_dataset(dataset, tokenizer, max_length) -> Dataset` — padding="max_length", truncation=True, set_format("torch")
- `load_dreaddit(config, tokenizer) -> tuple[DataLoader, DataLoader, DataLoader]` — StratifiedShuffleSplit for train/val, pin_memory=torch.cuda.is_available(), seeded Generator

### src/train.py

**Classes:**
- `TrainConfig` (dataclass): `lr: float = 2e-5`, `num_epochs: int = 30`, `warmup_fraction: float = 0.1`, `max_grad_norm: float = 1.0`, `patience: int = 3`, `seed: int = 42`, `checkpoint_dir: str = "results/checkpoints"`, `run_name: str = "default"`, `wandb_project: str = "lora-hopfield-clinical"`, `wandb_enabled: bool = True`, `num_labels: int = 2`

**Functions:**
- `seed_everything(seed: int) -> None` — sets random, np.random, torch.manual_seed, torch.cuda.manual_seed_all, cudnn.deterministic=True, cudnn.benchmark=False
- `_get_linear_warmup_cosine_scheduler(optimizer, num_warmup_steps, num_training_steps) -> LambdaLR`
  - Warmup: `current_step / max(1, num_warmup_steps)` (linear 0→1)
  - Decay: `0.5 * (1.0 + cos(π * progress))` where `progress = (step - warmup) / (total - warmup)`
  - warmup_steps = `int(total_steps * warmup_fraction)` = `int(len(train_loader) * num_epochs * 0.1)`
- `extract_last_hidden_state(hidden_states, attention_mask) -> torch.Tensor` — `attention_mask.sum(dim=-1) - 1` to find last real token index
- `_evaluate_epoch(model, head, loader, criterion, device) -> tuple[float, dict]` — @torch.no_grad()
- `train(model, train_loader, val_loader, test_loader, config, hidden_size=768) -> dict`
  - Classification head: `nn.Linear(hidden_size, num_labels)`
  - Optimizer: `AdamW([p for p in model.parameters() if p.requires_grad] + list(head.parameters()), lr=config.lr)`
  - Gradient clipping: `nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)`
  - Early stopping: patience=3 on val_loss
  - Checkpoint: saves best model + head state_dict
  - W&B: logs per-step loss/lr, per-epoch val metrics, final test metrics

### src/evaluate.py

**Functions:**
- `compute_metrics(y_true, y_pred, label_names=("not_stressed", "stressed")) -> dict`
  - Returns: accuracy, f1_macro, confusion_matrix (list[list[int]]), per_class (dict with precision/recall/f1/support)
  - All values JSON-serializable (no numpy types)

---

## 4. Key Architectural Numbers

| Parameter | Value | Source |
|---|---|---|
| GPT-2 small total parameters | 124,439,808 | model.config.n_embd, verified from result JSONs (124,441,346 includes cls head 768*2+2=1,538) |
| Number of attention blocks | 12 | model.config.n_layer |
| Hidden size | 768 | model.config.n_embd |
| Number of attention heads | 12 | model.config.n_head |
| d_head | 64 | 768 / 12 |
| c_attn projection shape | 768 → 2304 | Combined QKV: 768 → 3*768 |
| Classification head | 768 → 2 | nn.Linear(768, 2) = 1,538 params (768*2 + 2) |

### Trainable Parameter Counts per LoRA Rank (c_attn only, 12 blocks)

Formula per block: `r * (in_features + out_features)` = `r * (768 + 2304)` = `r * 3072`
Total across 12 blocks: `12 * r * 3072` = `36,864 * r`
Plus classification head: `+ 1,538`

| Rank | Per-Block Adapter | 12 Blocks | + Cls Head | Result JSON Match |
|---|---|---|---|---|
| r=1 | 3,072 | 36,864 | 38,402 | 38,402 ✓ |
| r=4 | 12,288 | 147,456 | 148,994 | 148,994 ✓ |
| r=8 | 24,576 | 294,912 | 296,450 | 296,450 ✓ |
| r=16 | 49,152 | 589,824 | 591,362 | 591,362 ✓ |
| r=32 | 98,304 | 1,179,648 | 1,181,186 | 1,181,186 ✓ |
| r=64 | 196,608 | 2,359,296 | 2,360,834 | 2,360,834 ✓ |
| r=128 | 393,216 | 4,718,592 | 4,720,130 | 4,720,130 ✓ |

All counts verified against result JSONs. Formula is exact.

---

## 5. Git Log — Commit History

| Hash | Message |
|---|---|
| da934d9 | Day 18: full fine-tune complete — lr=5e-6 rerun valid, test_acc=0.7664 matches LoRA r=32 |
| ede0818 | Day 18 prep: full fine-tune script + fp16 ROCm investigation (fp32 wins) |
| cf1d4fe | Day 17: Hopfield injection complete — test_acc=0.6811, test_f1=0.6772, hopfield-s42 |
| 4006018 | Day 16: LoRA rank ablation complete (r=1..128, early stopping, seed=42) |
| be1340e | feat(model): GPT-2 loading, Conv1D->Linear conversion, LoRA injection |
| 85d9c7a | feat(day14): data pipeline, training loop, evaluation metrics |
| 3fc94a1 | feat(hopfield): implement HopfieldAttention with beta-controlled retrieval per Ramsauer et al. |
| d97b0bb | chore: add .gitignore and project scaffolding |
| f7b9e76 | feat(lora): implement LoRALinear with zero-init B per Hu et al. Section 4 |

**Total commits:** 9
**Timeline:** Day 12 (lora.py) → Day 13 (hopfield.py) → Day 14 (data/train/eval) → Day 15 (model.py) → Day 16 (ablation) → Day 17 (Hopfield training) → Day 18 (full fine-tune + fp16 investigation)

---

## 6. Benchmarks — fp16 ROCm Investigation

**File:** `benchmarks/fp16_rocm_investigation.py`
**Results:** `benchmarks/results/fp16_rocm_results.md`

**Setup:** AMD Radeon RX 6700S (gfx1032 masquerading as gfx1030), ROCm 6.2.0, PyTorch 2.5.1+rocm6.2, hipBLASLt unsupported (falls back to hipBLAS). Benchmark: 20 training steps, Hopfield GPT-2 on DREADDIT.

| Config | Steps/s | Samples/s | Peak VRAM (MB) | Inf Gradient Steps |
|---|---|---|---|---|
| fp32, bs=16 | 4.65 | 74.5 | 741 | 0/20 |
| fp16, bs=16 | 6.09 | 97.4 | 687 | 6/20 |
| fp16, bs=32 | 2.95 | 94.3 | 898 | 6/20 |

**Key findings:**
- fp16 is 31% faster in raw throughput but 30% of steps produce Inf gradients
- GradScaler collapsed from 65536 → 256 in 26 steps — chronic instability, not transient warmup
- Effective throughput: fp16 × 0.70 = 68.2 samples/sec vs fp32 74.5 samples/sec → **fp32 is 9% faster in usable gradient updates**
- VRAM savings negligible (54 MB)
- **Decision:** fp32 only. fp16 rejected.

**Root cause:** gfx1032→gfx1030 masquerade disables hipBLASLt. The hipBLAS fallback path has less robust fp16 overflow handling than NVIDIA Tensor Cores. Known limitation of RDNA 2 under ROCm.

---

## 7. W&B Run ID Mapping

### Primary Ablation

| Run Name | W&B Run ID |
|---|---|
| frozen-baseline-s42 | yq5x52zv |
| lora-r1-s42 | vuykn55l |
| lora-r4-s42 | 348htjeh |
| lora-r8-s42 | 3qh60b02 |
| lora-r16-s42 | g8mkyemx |
| lora-r32-s42 | pq2plsoj |
| lora-r64-s42 | 0pob6ww7 |
| lora-r128-s42 | ggr67mmv |

### Epoch-10 Ablation

| Run Name | W&B Run ID |
|---|---|
| frozen-baseline (10ep) | 97u6rrst |
| lora-r1 (10ep) | tjzfzj67 |
| lora-r4 (10ep) | mz14fl4y |
| lora-r8 (10ep) | dugh977c |
| lora-r16 (10ep) | ivaw4khc |
| lora-r32 (10ep) | 4b91vkz9 |
| lora-r64 (10ep) | wcq4d1i7 |
| lora-r128 (10ep) | dq99fwc0 |

### Special Runs

| Run Name | W&B Run ID |
|---|---|
| hopfield-s42 | 3fx5qwm1 |
| full-finetune-s42 (lr=2e-5, bad) | 7ex5isv2 |
| full-finetune-s42 (lr=5e-6, canonical) | 5mocwj86 |

**Total W&B runs:** 19

---

## 8. CLAUDE.md — Explicit Design Rules

### NEVER DO Rules

- Never hardcode `"cuda"` — always use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Never use `.T` on 3D+ tensors — use `.transpose(-2, -1)` (`.T` transposes ALL dims)
- Never implement raw exp/sum for softmax — use `F.softmax(..., dim=-1)`
- Never use `# TODO: implement this` or `pass` as function body in committed code
- Never use `git add .` — stage files individually
- Never write training loops without explicit seed setting
- Never write a module without a corresponding test file
- Never name variables `result`, `output`, `data`, `temp` without qualification
- Never write comments that restate code (`# multiply A and B`)
- Never use bare `assert` — use `raise ValueError(...)` with context
- Never use `Dict[str, int]` — use `dict[str, int]` (Python 3.11 style)
- Never use `Optional[X]` — use `X | None`

### ALWAYS DO Rules

- Every function: complete type annotations on all parameters and return types
- Every class/public method: docstring with math notation, args with shapes, paper references
- All hyperparameters in dataclasses, never as magic numbers
- `register_buffer` for non-trainable tensors that should move with `.to(device)`
- `nn.Parameter` for learnable tensors not managed by submodules
- Gradient tests for every new `nn.Module`: verify gradients flow to trainable params, not to frozen
- Every file has one job; split at ~300 lines

### Explicit Gotchas Documented

1. **Conv1D vs nn.Linear:** HuggingFace GPT-2 uses `Conv1D` (from `transformers.pytorch_utils`), NOT `nn.Linear`. Conv1D stores weights transposed: `(in_features, out_features)` vs nn.Linear's `(out_features, in_features)`. Must convert.

2. **GPT-2 attention class:** GPT-2 does NOT use `nn.MultiheadAttention`. It uses a bespoke `GPT2Attention` class with Conv1D projections and manual QKV splitting. Replacement targets the attention weight computation, not an nn.MultiheadAttention.

3. **GPT2Model vs GPT2LMHeadModel:** Use `GPT2Model` (base transformer) — `.last_hidden_state` exists here. `GPT2LMHeadModel` returns `.logits` instead.

4. **c_attn is combined QKV:** Shape 768→2304 for gpt2-small (3×768). Split after projection.

5. **Hopfield overflow:** `exp(beta * similarity)` overflows with large beta or high-dim embeddings. Use `F.softmax` for built-in log-sum-exp stability.

6. **Dataset path:** `andreagasparini/dreaddit` (NOT `dkmkknub/dreaddit` — dead path).

### WRONG/RIGHT Code Examples in CLAUDE.md

| Pattern | WRONG | RIGHT |
|---|---|---|
| Type annotations | `def forward(self, x):` | `def forward(self, x: torch.Tensor) -> torch.Tensor:` |
| Docstrings | `"""LoRA linear layer."""` | Full docstring with math, args, shapes, paper refs |
| Buffers | `beta = torch.ones(n) * val` in forward | `register_buffer("beta", torch.tensor(beta))` in `__init__` |
| Validation | `assert r > 0` | `if r <= 0: raise ValueError(f"LoRA rank r must be positive, got r={r}...")` |
| Transpose | `.T` on batched tensor | `.transpose(-2, -1)` |
| Typing | `Dict[str, int]`, `Optional[X]` | `dict[str, int]`, `X \| None` |

---

## 9. Test Class Inventory

### tests/test_lora.py — 31 tests, 8 classes

| Class | Tests |
|---|---|
| TestLoRAConfig | 2 (test_defaults, test_custom_values) |
| TestLoRALinearInit | 7 (test_zero_init_B, test_A_is_not_zero, test_base_weight_frozen, test_base_bias_frozen, test_no_bias, test_adapter_shapes, test_scaling_factor) |
| TestLoRALinearValidation | 3 (test_rank_zero_raises, test_rank_negative_raises, test_rank_exceeds_dimensions_raises) |
| TestLoRALinearForward | 5 (test_output_shape_2d, test_output_shape_3d, test_output_shape_4d, test_identity_at_init, test_adapter_contributes_after_update) |
| TestLoRALinearGradients | 4 (test_gradients_flow_to_adapter, test_no_gradients_to_frozen_weight, test_no_gradients_to_frozen_bias, test_gradient_magnitude_scales_with_alpha) |
| TestLoRALinearParameterCount | 3 (test_trainable_params_no_bias, test_trainable_params_with_bias, test_trainable_params_various_ranks) |
| TestLoRALinearFromLinear | 5 (test_weight_copy, test_bias_copy, test_from_linear_preserves_output, test_from_linear_no_bias, test_from_linear_does_not_share_memory) |
| TestLoRALinearDropout | 2 (test_dropout_active_in_train_mode, test_dropout_inactive_in_eval_mode) |

### tests/test_hopfield.py — 30 tests, 11 classes

| Class | Tests |
|---|---|
| TestHopfieldConfig | 2 (test_defaults, test_custom_values) |
| TestHopfieldValidation | 5 (test_d_head_zero_raises, test_d_head_negative_raises, test_num_iters_zero_raises, test_beta_zero_raises, test_beta_negative_raises) |
| TestHopfieldShapes | 3 (test_output_shape_square, test_output_shape_cross_attention, test_output_shape_single_token) |
| TestStandardAttentionRecovery | 3 (test_matches_scaled_dot_product_no_mask, test_matches_scaled_dot_product_with_causal_mask, test_explicit_beta_matches_default) |
| TestCausalMasking | 2 (test_first_token_attends_only_to_self, test_causal_changes_output) |
| TestBetaBehavior | 3 (test_high_beta_sharpens_attention, test_low_beta_softens_attention, test_different_beta_different_output) |
| TestMultiIteration | 2 (test_single_vs_multi_differ, test_convergence) |
| TestNumericalStability | 3 (test_large_beta_no_nan, test_small_beta_no_nan, test_large_values_no_overflow) |
| TestGradientFlow | 3 (test_gradients_to_qkv, test_gradients_with_causal_mask, test_gradients_multi_iter) |
| TestBufferBehavior | 3 (test_beta_is_buffer_not_parameter, test_no_learnable_parameters, test_default_beta_value) |
| TestRepr | 1 (test_extra_repr_contains_config) |

### tests/test_hopfield_gpt2.py — 17 tests, 7 classes

| Class | Tests |
|---|---|
| TestInjectHopfield | 6 (test_all_blocks_have_hopfield, test_all_base_params_frozen, test_returns_12_blocks, test_zero_trainable_params, test_beta_buffer_on_each_block, test_custom_config_propagated) |
| TestBetaRecovery | 2 (test_identical_output_default_beta, test_identical_output_with_attention_mask) |
| TestOutputShape | 3 (test_last_hidden_state_shape, test_variable_sequence_length, test_single_token_sequence) |
| TestBetaEffect | 1 (test_different_beta_gives_different_output) |
| TestVerifyHopfieldInjection | 2 (test_passes_on_correct_injection, test_fails_on_uninjected_model) |
| TestGradientFlow | 2 (test_backward_pass_succeeds, test_head_params_trainable_model_params_frozen) |
| TestDeviceMovement | 1 (test_beta_moves_to_device) |

### tests/test_data.py — 9 tests, 2 classes

| Class | Tests |
|---|---|
| TestTokenizeDataset | 8 (test_output_columns, test_original_columns_removed, test_output_length_matches_input, test_max_length_enforced, test_torch_format, test_labels_preserved, test_padding_creates_attention_mask, test_pad_token_id_in_padded_positions) |
| TestDataConfig | 1 (test_defaults) |

### tests/test_model.py — 26 tests, 8 classes

| Class | Tests |
|---|---|
| TestConv1dToLinear | 5 (test_output_equivalence_square, test_output_equivalence_rectangular, test_weight_shapes, test_does_not_share_memory, test_bias_copied) |
| TestLoadGpt2 | 5 (test_returns_gpt2model, test_has_last_hidden_state, test_num_blocks, test_hidden_size, test_attention_layers_are_conv1d) |
| TestFreezeUnfreeze | 2 (test_freeze_all, test_unfreeze_all) |
| TestInjectLora | 8 (test_replaces_c_attn, test_replaces_c_attn_and_c_proj, test_c_proj_not_replaced_when_not_targeted, test_trainable_param_count_c_attn, test_trainable_param_count_various_ranks, test_base_weights_frozen_after_injection, test_invalid_target_raises, test_output_shape_preserved) |
| TestOutputEquivalenceAtInit | 1 (test_identical_output_at_init) |
| TestVerifyLoraInjection | 2 (test_passes_on_correct_injection, test_passes_with_c_attn_and_c_proj) |
| TestFrozenBaseline | 2 (test_zero_trainable_params, test_still_produces_output) |
| TestFullFineTune | 1 (test_all_params_trainable) |

### tests/test_train.py — 14 tests, 5 classes

| Class | Tests |
|---|---|
| TestSeedEverything | 2 (test_torch_reproducible, test_different_seeds_differ) |
| TestExtractLastHiddenState | 4 (test_no_padding, test_with_padding, test_single_token, test_variable_lengths) |
| TestLRScheduler | 4 (test_warmup_starts_at_zero, test_warmup_reaches_peak, test_cosine_decays_to_zero, test_warmup_is_monotonically_increasing) |
| TestOptimizerFiltering | 2 (test_filter_excludes_frozen_params, test_frozen_params_unchanged_after_step) |
| TestTrainConfig | 2 (test_defaults) |

### tests/test_evaluate.py — 10 tests, 4 classes

| Class | Tests |
|---|---|
| TestComputeMetricsBasic | 3 (test_perfect_predictions, test_all_wrong, test_mixed_predictions) |
| TestComputeMetricsPerClass | 3 (test_per_class_keys_present, test_per_class_values_sum_to_total, test_custom_label_names) |
| TestComputeMetricsValidation | 3 (test_empty_arrays_raise, test_length_mismatch_raises, test_accepts_python_lists) |
| TestJsonSerializable | 2 (test_json_round_trip, test_no_numpy_types_in_output) |

### Grand Total

| Metric | Count |
|---|---|
| Test files | 7 |
| Test classes | 45 |
| Test methods | 137 |
| All passing | Yes |
