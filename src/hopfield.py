"""
Hopfield Networks is All You Need (Ramsauer et al., 2020).

This module implements the modern continuous Hopfield retrieval rule as an
attention mechanism. The core theorem from Ramsauer et al.: transformer
scaled dot-product attention is exactly one step of energy minimization in
a continuous Hopfield network with exponential interaction function.

The energy being minimized:
    E(ξ) = -β⁻¹ log Σ_μ exp(β ξ · x_μ^T) + ½ ||ξ||²

The retrieval (update) rule that minimizes this energy:
    ξ_new = softmax(β · ξ · X^T) · X

When β = 1/√d_head, this recovers standard scaled dot-product attention:
    Attn(Q, K, V) = softmax(QK^T / √d) V

This connection matters for Phase 1 context: the HyDE experiment showed that
retrieval quality depends on how attention resolves ambiguous patterns (informal
Reddit language vs. clinical vocabulary). Hopfield theory explains *why*:
β controls pattern separation — high β produces winner-take-all retrieval
(sharp, precise matches), low β produces distributed retrieval (soft, blended
matches). The Phase 1 finding that large chunks helped HyDE maps to the
Hopfield picture: more stored patterns with overlapping content lower the
energy barrier, making retrieval more robust to vocabulary mismatch.

Reference: https://arxiv.org/abs/2008.02217
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HopfieldConfig:
    """Configuration for Hopfield attention replacement.

    Attributes:
        beta: Inverse temperature for energy-based retrieval. Controls pattern
            separation — high β gives sharp, winner-take-all attention; low β
            gives soft, distributed attention. When None, defaults to
            1/√d_head (recovering standard scaled dot-product attention).
        num_iters: Number of Hopfield retrieval iterations. One iteration
            equals standard attention. Multiple iterations converge toward
            the energy minimum (a fixed point / metastable state). Default 1.
        dropout: Dropout on attention weights after softmax. Matches the
            regularization used in GPT-2's native attention.
    """

    beta: float | None = None
    num_iters: int = 1
    dropout: float = 0.0


def hopfield_retrieval(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    causal_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Single step of Hopfield energy minimization (the retrieval rule).

    Computes: ξ_new = softmax(β · Q · K^T) · V

    This is a pure function (no learned parameters) for testability. The
    HopfieldAttention module wraps this with configuration and state.

    In Hopfield terminology:
        - queries (ξ): state patterns — what we want to retrieve/complete
        - keys (X): stored patterns — the memory bank to retrieve from
        - values: the content associated with each stored pattern
        - beta (β): inverse temperature controlling pattern separation

    Args:
        queries: Query tensor, shape (B, H, N, d_head).
        keys: Key tensor, shape (B, H, M, d_head).
        values: Value tensor, shape (B, H, M, d_head).
        beta: Inverse temperature scalar, shape () — registered buffer.
        causal_mask: Boolean mask, shape (1, 1, N, M) or broadcastable.
            True at positions that should be MASKED (i.e., future tokens).
            Converted to additive -inf before softmax.
        dropout_p: Dropout probability on attention weights (applied only
            during training).
        training: Whether the model is in training mode (controls dropout).

    Returns:
        Retrieved patterns, shape (B, H, N, d_head).
    """
    # β · Q · K^T — the Hopfield energy interaction term
    # .transpose(-2, -1) swaps (M, d_head) -> (d_head, M); NOT .T which
    # would transpose all dims on a 4D tensor
    scores = beta * (queries @ keys.transpose(-2, -1))  # (B, H, N, M)

    # Causal masking: set future positions to -inf before softmax so they
    # receive zero attention weight. Must happen before softmax, not after —
    # masking after softmax would allow information leakage from future tokens.
    if causal_mask is not None:
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # F.softmax handles the log-sum-exp trick internally: it subtracts the
    # row-wise max before exponentiation, preventing overflow even at large β.
    # This is the critical stability guarantee from Ramsauer et al.'s appendix.
    attn_weights = F.softmax(scores, dim=-1)  # (B, H, N, M)

    if dropout_p > 0.0 and training:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    # Retrieval: weight the stored values by attention — the energy-minimizing
    # associative memory readout
    return attn_weights @ values  # (B, H, N, d_head)


class HopfieldAttention(nn.Module):
    """Energy-based attention via modern continuous Hopfield retrieval.

    Drop-in replacement for the core attention computation inside GPT-2's
    GPT2Attention class. Replaces the softmax(QK^T/√d) @ V step with the
    Hopfield update rule: softmax(β · Q · K^T) · V.

    This does NOT replace the QKV projections or output projection — only
    the attention weight computation between them. GPT-2 uses bespoke
    Conv1D layers for projections; those remain untouched.

    When beta is None (default), uses β = 1/√d_head, which exactly recovers
    standard scaled dot-product attention. Explicit β enables exploration of
    the Hopfield energy landscape — higher β sharpens attention (better
    pattern discrimination for out-of-distribution inputs like Reddit slang),
    lower β blurs it (more robust to noise but less precise).

    Args:
        d_head: Dimension of each attention head. Used to compute default
            β = 1/√d_head. For GPT-2 small: d_head = 768/12 = 64.
        config: HopfieldConfig controlling beta, iteration count, dropout.

    Example:
        >>> attn = HopfieldAttention(d_head=64)
        >>> q = k = v = torch.randn(2, 12, 10, 64)
        >>> out = attn(q, k, v)  # shape: (2, 12, 10, 64)
    """

    def __init__(self, d_head: int, config: HopfieldConfig | None = None) -> None:
        super().__init__()
        config = config or HopfieldConfig()

        if d_head <= 0:
            raise ValueError(
                f"d_head must be positive, got d_head={d_head}. "
                "For GPT-2 small: d_head = 768 / 12 = 64."
            )
        if config.num_iters < 1:
            raise ValueError(
                f"num_iters must be >= 1, got {config.num_iters}. "
                "One iteration recovers standard attention."
            )

        self.d_head = d_head
        self.num_iters = config.num_iters
        self.dropout_p = config.dropout

        # β as a buffer: not learned, but must move with .to(device).
        # Default β = 1/√d_head recovers standard scaled dot-product attention.
        beta_val = config.beta if config.beta is not None else 1.0 / (d_head ** 0.5)
        if beta_val <= 0.0:
            raise ValueError(
                f"beta must be positive, got beta={beta_val}. "
                "beta=0 would give uniform attention (degenerate); "
                "beta<0 would invert the energy landscape."
            )
        self.register_buffer("beta", torch.tensor(beta_val, dtype=torch.float32))

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run Hopfield retrieval (one or more iterations).

        Iteration 1: standard Hopfield retrieval (equivalent to attention).
        Iterations 2+: re-feed the retrieved output as the new query,
        converging toward the Hopfield energy minimum (fixed point).

        Args:
            queries: Shape (B, H, N, d_head) — state patterns to complete.
            keys: Shape (B, H, M, d_head) — stored memory patterns.
            values: Shape (B, H, M, d_head) — content to retrieve.
            causal_mask: Bool tensor, True at positions to mask (future tokens).
                Shape (1, 1, N, M) or broadcastable.

        Returns:
            Retrieved patterns, shape (B, H, N, d_head).
        """
        result = queries
        for _ in range(self.num_iters):
            result = hopfield_retrieval(
                queries=result,
                keys=keys,
                values=values,
                beta=self.beta,
                causal_mask=causal_mask,
                dropout_p=self.dropout_p,
                training=self.training,
            )
        return result

    def extra_repr(self) -> str:
        """Show Hopfield config in print(model)."""
        return (
            f"d_head={self.d_head}, beta={self.beta.item():.6f}, "
            f"num_iters={self.num_iters}, dropout={self.dropout_p}"
        )
