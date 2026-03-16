"""
GPT-2 loading, LoRA injection, and model configuration for DREADDIT classification.

This module is the integration layer that connects the LoRA adapter (src/lora.py)
to the pretrained GPT-2 backbone. It handles three configurations for the
Day 18 comparison table:
    1. Frozen baseline — all GPT-2 weights frozen, only classification head trains
    2. LoRA — base frozen, low-rank adapters injected into attention projections
    3. Full fine-tune — all weights trainable (upper bound reference)

GPT-2 model selection: we use GPT2Model (the base transformer), NOT
GPT2LMHeadModel or AutoModelForCausalLM. train.py accesses
outputs.last_hidden_state which only exists on GPT2Model's output type
(BaseModelOutputWithPastAndCrossAttentions). GPT2LMHeadModel returns
CausalLMOutputWithCrossAttentions which has .logits instead — using the
wrong class causes AttributeError at training time.

Conv1D gotcha: HuggingFace GPT-2 uses Conv1D (from transformers.pytorch_utils),
NOT nn.Linear, for all projection layers. Conv1D stores weight as (in, out) and
computes x @ weight + bias; nn.Linear stores weight as (out, in) and computes
x @ weight.T + bias. Same math, transposed storage. LoRALinear wraps nn.Linear,
so we must convert Conv1D -> nn.Linear (transposing the weight) before injection.

Phase 1 connection: this is where the LoRA story meets the RAG retrieval story.
The c_attn projection is the QKV computation that determines what each token
attends to — it's the mechanism that Phase 1's retrieval pipeline depended on.
LoRA adapts this projection cheaply, learning task-specific attention patterns
for informal Reddit text without overwriting the pretrained knowledge.

Reference: Hu et al. (2021), https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from transformers import GPT2Model
from transformers.pytorch_utils import Conv1D

from src.lora import LoRAConfig, LoRALinear

logger = logging.getLogger(__name__)


def conv1d_to_linear(conv1d: Conv1D) -> nn.Linear:
    """Convert a HuggingFace Conv1D layer to nn.Linear.

    Conv1D stores weight as (in_features, out_features) and computes
    x @ weight + bias. nn.Linear stores (out_features, in_features)
    and computes x @ weight.T + bias. The conversion transposes the
    weight so the two layers produce identical outputs.

    Args:
        conv1d: A transformers Conv1D layer with .weight (in, out),
            .bias (out,), and .nf (out_features).

    Returns:
        An nn.Linear with weight = conv1d.weight.T (cloned, not shared).
        Produces identical output to the original Conv1D for any input.
    """
    out_features = conv1d.nf
    in_features = conv1d.weight.shape[0]
    linear = nn.Linear(in_features, out_features, bias=conv1d.bias is not None)

    # Transpose: Conv1D (in, out) -> nn.Linear (out, in)
    # .contiguous() ensures the transposed tensor has its own memory layout
    linear.weight.data = conv1d.weight.data.T.contiguous()
    if conv1d.bias is not None:
        linear.bias.data = conv1d.bias.data.clone()

    return linear


def load_gpt2(model_name: str = "gpt2") -> GPT2Model:
    """Load the GPT-2 base transformer (not the LM head variant).

    Returns GPT2Model whose forward() produces outputs with
    .last_hidden_state of shape (B, seq_len, hidden_size). This is
    required by train.py's extract_last_hidden_state function.

    Args:
        model_name: HuggingFace model identifier. "gpt2" is the 124M
            small variant that fits in 8GB VRAM with LoRA overhead.

    Returns:
        GPT2Model in eval mode with all parameters requiring gradients
        (caller is responsible for freezing as needed).
    """
    model = GPT2Model.from_pretrained(model_name)
    logger.info(
        "Loaded %s: %d params, %d blocks, hidden_size=%d",
        model_name,
        sum(p.numel() for p in model.parameters()),
        len(model.h),
        model.config.hidden_size,
    )
    return model


def freeze_all_parameters(model: nn.Module) -> None:
    """Freeze every parameter in a model (requires_grad = False).

    Used for frozen baseline (no adapters) and as the first step of
    LoRA injection (freeze base, then attach trainable adapters).

    Args:
        model: Any nn.Module — all parameters recursively frozen.
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_all_parameters(model: nn.Module) -> None:
    """Unfreeze every parameter in a model (requires_grad = True).

    Used for the full fine-tune configuration (upper bound reference).

    Args:
        model: Any nn.Module — all parameters recursively unfrozen.
    """
    for param in model.parameters():
        param.requires_grad = True


def inject_lora(model: GPT2Model, config: LoRAConfig) -> dict[str, int]:
    """Inject LoRA adapters into a GPT-2 model's attention layers.

    Walks the model's transformer blocks, converts target Conv1D layers
    to nn.Linear, then wraps them with LoRALinear. All base weights are
    frozen first — only the adapter parameters (A, B) are trainable.

    The injection targets are specified by config.target_modules, which
    names the Conv1D submodules within each GPT2Attention block to replace.
    Default is ("c_attn",) — the combined QKV projection. Can also include
    "c_proj" for the attention output projection.

    This function must be standalone (not part of a class __init__) per
    the architecture plan, so it can be called on any loaded GPT2Model.

    Args:
        model: A GPT2Model (base transformer, not LM head). All its
            parameters will be frozen before adapter injection.
        config: LoRAConfig specifying rank, alpha, dropout, and target modules.

    Returns:
        Dict mapping replaced module paths to their trainable parameter count.
        Example: {"h.0.attn.c_attn": 12288, "h.1.attn.c_attn": 12288, ...}
    """
    freeze_all_parameters(model)

    replaced = {}

    for block_idx, block in enumerate(model.h):
        attn = block.attn

        for target_name in config.target_modules:
            if not hasattr(attn, target_name):
                raise ValueError(
                    f"GPT2Attention has no attribute '{target_name}'. "
                    f"Valid targets: c_attn, c_proj. Got config.target_modules={config.target_modules}"
                )

            conv1d_module = getattr(attn, target_name)
            if not isinstance(conv1d_module, Conv1D):
                # Already replaced (e.g., inject_lora called twice) or unexpected type
                logger.warning(
                    "h.%d.attn.%s is %s, not Conv1D — skipping",
                    block_idx,
                    target_name,
                    type(conv1d_module).__name__,
                )
                continue

            # Conv1D -> nn.Linear -> LoRALinear
            linear = conv1d_to_linear(conv1d_module)
            lora_layer = LoRALinear.from_linear(
                linear, r=config.r, alpha=config.alpha, dropout=config.dropout
            )

            setattr(attn, target_name, lora_layer)

            module_path = f"h.{block_idx}.attn.{target_name}"
            trainable_count = sum(
                p.numel() for p in lora_layer.parameters() if p.requires_grad
            )
            replaced[module_path] = trainable_count

            logger.debug(
                "Replaced %s: Conv1D(%d, %d) -> LoRALinear(r=%d, trainable=%d)",
                module_path,
                linear.in_features,
                linear.out_features,
                config.r,
                trainable_count,
            )

    total_trainable = sum(replaced.values())
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA injection complete: %d modules replaced, %d trainable params (%.4f%% of %d total)",
        len(replaced),
        total_trainable,
        100.0 * total_trainable / total_params,
        total_params,
    )

    return replaced


def verify_lora_injection(model: GPT2Model, config: LoRAConfig) -> dict[str, object]:
    """Post-injection diagnostic confirming correct LoRA setup.

    Checks three invariants:
    1. Each replaced module has exactly r*(in+out) trainable params (A + B)
    2. All base weights (frozen linear, embeddings, LayerNorm) have requires_grad=False
    3. All LoRA A and B parameters have requires_grad=True

    Args:
        model: GPT2Model after inject_lora() has been called.
        config: The LoRAConfig used for injection.

    Returns:
        Summary dict with:
            - total_params: int
            - trainable_params: int
            - trainable_pct: float
            - lora_modules: list of replaced module paths
            - all_checks_passed: bool

    Raises:
        AssertionError: If any invariant is violated, with a descriptive message.
    """
    lora_modules = []
    all_passed = True

    for block_idx, block in enumerate(model.h):
        attn = block.attn
        for target_name in config.target_modules:
            module_path = f"h.{block_idx}.attn.{target_name}"
            module = getattr(attn, target_name, None)

            if not isinstance(module, LoRALinear):
                logger.error("%s is not LoRALinear (got %s)", module_path, type(module).__name__)
                all_passed = False
                continue

            lora_modules.append(module_path)

            # Check 1: trainable param count = r * (in + out)
            expected_trainable = config.r * (module.in_features + module.out_features)
            actual_trainable = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            if actual_trainable != expected_trainable:
                logger.error(
                    "%s: expected %d trainable params, got %d",
                    module_path,
                    expected_trainable,
                    actual_trainable,
                )
                all_passed = False

            # Check 2: base weight is frozen
            if module.linear.weight.requires_grad:
                logger.error("%s: base weight is NOT frozen", module_path)
                all_passed = False
            if module.linear.bias is not None and module.linear.bias.requires_grad:
                logger.error("%s: base bias is NOT frozen", module_path)
                all_passed = False

            # Check 3: adapter params are trainable
            if not module.lora_A.requires_grad:
                logger.error("%s: lora_A is NOT trainable", module_path)
                all_passed = False
            if not module.lora_B.requires_grad:
                logger.error("%s: lora_B is NOT trainable", module_path)
                all_passed = False

    # Check non-LoRA parameters are all frozen
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            continue
        if param.requires_grad:
            logger.error("Non-adapter parameter is trainable: %s", name)
            all_passed = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    summary = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": 100.0 * trainable_params / total_params,
        "lora_modules": lora_modules,
        "all_checks_passed": all_passed,
    }

    if all_passed:
        logger.info(
            "Verification passed: %d LoRA modules, %d trainable params (%.4f%%)",
            len(lora_modules),
            trainable_params,
            summary["trainable_pct"],
        )
    else:
        logger.error("Verification FAILED — see errors above")

    return summary


def print_param_table(model: nn.Module, head: nn.Module | None = None) -> None:
    """Print a summary table of total, trainable, and frozen parameters.

    Includes breakdown by module type for readability in experiment logs.

    Args:
        model: The GPT-2 model (with or without LoRA).
        head: Optional classification head to include in the count.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    head_params = 0
    if head is not None:
        head_params = sum(p.numel() for p in head.parameters())
        total += head_params
        trainable += head_params

    print(f"{'Parameter Summary':=^50}")
    print(f"  Total params:     {total:>12,}")
    print(f"  Trainable params: {trainable:>12,}")
    print(f"  Frozen params:    {frozen:>12,}")
    if head is not None:
        print(f"  Head params:      {head_params:>12,}")
    print(f"  Trainable %:      {100.0 * trainable / total:>11.4f}%")
    print(f"{'':=^50}")
