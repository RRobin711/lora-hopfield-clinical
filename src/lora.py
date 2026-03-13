"""
LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021).

This module implements the core LoRALinear layer — a frozen nn.Linear with a
trainable low-rank bypass. The motivation: transformer attention is the
mechanism Phase 1 (Clinical RAG) relied on for semantic retrieval, and LoRA
is how you adapt that mechanism cheaply to new domains (e.g., clinical text,
informal Reddit language in DREADDIT) without full fine-tuning cost.

The key insight from Hu et al.: pretrained weight matrices are "full rank" but
the *task-specific adaptation* lives in a much lower-dimensional subspace.
LoRA exploits this by decomposing the weight update into two small matrices:
    delta_W = B @ A    where A: (r, d_in), B: (d_out, r), r << min(d_in, d_out)

This gives the same expressiveness for the adaptation while training only
r * (d_in + d_out) parameters instead of d_in * d_out.

Reference: https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapter injection.

    Controls rank, scaling, dropout, and which modules to target when
    injecting adapters into a pretrained model.

    Attributes:
        r: LoRA rank — dimensionality of the low-rank decomposition.
            Paper ablates r in {1, 2, 4, 8, 16, 64}. r=4 recovers most
            full fine-tune quality; higher ranks yield diminishing returns.
        alpha: Scaling factor. The adapter output is scaled by alpha/r,
            so setting alpha=r gives scale=1.0 and makes the effective
            learning rate invariant to the rank choice.
        dropout: Dropout probability applied to input before the low-rank
            projection. Regularizes the adapter path without affecting the
            frozen base weight path.
        target_modules: Names of modules to replace with LoRA-wrapped versions.
            For GPT-2: "c_attn" is the combined QKV projection in each block.
    """

    r: int = 4
    alpha: float = 4.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("c_attn",)


class LoRALinear(nn.Module):
    """Low-rank adaptation of a frozen nn.Linear layer.

    Implements the LoRA reparameterization from Hu et al. (2021), Section 4:
        h = W_0 x + (B A) x * (alpha / r)
    where W_0 is frozen, A is initialized with Kaiming uniform, B is zero-initialized
    so the adapter contributes nothing at the start of training.

    This zero-init of B is critical: it means the model starts from exactly the
    pretrained checkpoint, not a random perturbation of it. The training signal
    must "earn" every deviation from the pretrained weights.

    The scaling factor alpha/r decouples rank from learning rate: changing r
    during ablation doesn't require re-tuning the optimizer lr. This is why
    the Phase 2 rank ablation (r in {1, 4, 8, 16, 32}) can use a single lr
    across all configurations.

    Args:
        in_features: Input dimension d_in.
        out_features: Output dimension d_out.
        r: LoRA rank. Must satisfy 0 < r <= min(in_features, out_features).
        alpha: Scaling factor. Effective adapter scale = alpha / r.
        dropout: Dropout probability on input before the low-rank path.
        bias: Whether the base linear layer includes a bias term.

    Example:
        >>> layer = LoRALinear(768, 768, r=4, alpha=4.0)
        >>> x = torch.randn(2, 10, 768)
        >>> out = layer(x)  # shape: (2, 10, 768)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if r <= 0:
            raise ValueError(
                f"LoRA rank r must be positive, got r={r}. "
                "Common values: r=4 (efficient), r=16 (near full fine-tune quality)."
            )
        if r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank r={r} exceeds min(in_features={in_features}, "
                f"out_features={out_features})={min(in_features, out_features)}. "
                "The decomposition would not be low-rank."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Frozen base weight — the pretrained linear transformation
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # Low-rank adapter matrices
        # A: projects input down to rank-r subspace
        # B: projects rank-r representation back up to output dimension
        # B is zero-init so BA = 0 at t=0 — adapter is identity at initialization
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Kaiming uniform for A — matches the default nn.Linear init, appropriate
        # because A is the "input projection" of the adapter
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Dropout on input before the low-rank path only — does not affect W_0 x
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frozen linear + scaled low-rank adapter.

        Computes: h = W_0 x + (B @ A @ dropout(x)) * (alpha / r)

        The frozen path (W_0 x) is always computed. The adapter path (BA x)
        starts at zero and learns task-specific adjustments during fine-tuning.

        Args:
            x: Input tensor of shape (*, in_features) where * is any number
                of leading dimensions (batch, sequence, etc.).

        Returns:
            Output tensor of shape (*, out_features).
        """
        # Frozen pretrained path
        base_out = self.linear(x)

        # Low-rank adapter path: dropout applied to x before projection
        # F.linear computes x @ weight.T (+ bias), so lora_A: (r, in) gives
        # intermediate shape (*, r), then lora_B: (out, r) gives (*, out)
        adapter_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)

        return base_out + adapter_out * self.scaling

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.0,
    ) -> LoRALinear:
        """Wrap an existing nn.Linear with a LoRA adapter.

        Copies the pretrained weight (and bias) into the frozen base layer,
        then attaches a fresh low-rank adapter. Used during model injection
        to replace target modules in-place.

        Args:
            linear: The pretrained nn.Linear to wrap. Its weight and bias are
                copied, not shared — the original can be safely discarded.
            r: LoRA rank.
            alpha: Scaling factor.
            dropout: Dropout probability for the adapter path.

        Returns:
            A LoRALinear with the pretrained weight frozen and a zero-initialized
            adapter ready for fine-tuning.
        """
        has_bias = linear.bias is not None
        lora_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=has_bias,
        )

        # Copy pretrained weights into the frozen base layer
        lora_layer.linear.weight.data.copy_(linear.weight.data)
        if has_bias:
            lora_layer.linear.bias.data.copy_(linear.bias.data)

        return lora_layer

    def extra_repr(self) -> str:
        """Informative repr for print(model) — shows LoRA config alongside base layer."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"bias={self.linear.bias is not None}"
        )
