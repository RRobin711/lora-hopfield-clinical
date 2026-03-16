"""Tests for HopfieldAttention module.

Verifies the core invariants from Ramsauer et al. (2020):
- Standard attention recovery: at β = 1/√d_head, output matches scaled dot-product
- Shape preservation: output shape equals query shape
- Causal masking: future tokens receive zero attention weight
- Numerical stability: no NaN/inf at large β values
- Gradient flow: gradients propagate through to Q, K, V inputs
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.hopfield import HopfieldAttention, HopfieldConfig, hopfield_retrieval


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH, HEADS, SEQ, D_HEAD = 2, 4, 8, 64


def _make_qkv(
    seq_len: int = SEQ,
    d_head: int = D_HEAD,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random Q, K, V tensors with reproducible seed."""
    torch.manual_seed(42)
    q = torch.randn(BATCH, HEADS, seq_len, d_head)
    k = torch.randn(BATCH, HEADS, seq_len, d_head)
    v = torch.randn(BATCH, HEADS, seq_len, d_head)
    return q, k, v


def _causal_mask(seq_len: int) -> torch.Tensor:
    """Upper-triangular causal mask: True where attention is forbidden (j > i)."""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestHopfieldConfig:
    """Verify config dataclass defaults."""

    def test_defaults(self) -> None:
        cfg = HopfieldConfig()
        assert cfg.beta is None
        assert cfg.num_iters == 1
        assert cfg.dropout == 0.0

    def test_custom_values(self) -> None:
        cfg = HopfieldConfig(beta=0.5, num_iters=3, dropout=0.1)
        assert cfg.beta == 0.5
        assert cfg.num_iters == 3
        assert cfg.dropout == 0.1


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestHopfieldValidation:
    """Ensure invalid configurations raise clear errors."""

    def test_d_head_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="d_head must be positive"):
            HopfieldAttention(d_head=0)

    def test_d_head_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="d_head must be positive"):
            HopfieldAttention(d_head=-1)

    def test_num_iters_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_iters must be >= 1"):
            HopfieldAttention(d_head=64, config=HopfieldConfig(num_iters=0))

    def test_beta_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="beta must be positive"):
            HopfieldAttention(d_head=64, config=HopfieldConfig(beta=0.0))

    def test_beta_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="beta must be positive"):
            HopfieldAttention(d_head=64, config=HopfieldConfig(beta=-1.0))


# ---------------------------------------------------------------------------
# Shape preservation tests
# ---------------------------------------------------------------------------


class TestHopfieldShapes:
    """Output shape must match query shape for any sequence length."""

    def test_output_shape_square(self) -> None:
        """Q, K, V all same sequence length."""
        attn = HopfieldAttention(d_head=D_HEAD)
        q, k, v = _make_qkv()
        out = attn(q, k, v)
        assert out.shape == q.shape

    def test_output_shape_cross_attention(self) -> None:
        """K, V have different sequence length than Q (cross-attention)."""
        attn = HopfieldAttention(d_head=D_HEAD)
        torch.manual_seed(42)
        q = torch.randn(BATCH, HEADS, 5, D_HEAD)
        k = torch.randn(BATCH, HEADS, 12, D_HEAD)
        v = torch.randn(BATCH, HEADS, 12, D_HEAD)
        out = attn(q, k, v)
        assert out.shape == (BATCH, HEADS, 5, D_HEAD)

    def test_output_shape_single_token(self) -> None:
        """Edge case: single-token sequence."""
        attn = HopfieldAttention(d_head=D_HEAD)
        q, k, v = _make_qkv(seq_len=1)
        out = attn(q, k, v)
        assert out.shape == (BATCH, HEADS, 1, D_HEAD)


# ---------------------------------------------------------------------------
# Standard attention recovery — the core theoretical invariant
# ---------------------------------------------------------------------------


class TestStandardAttentionRecovery:
    """At β = 1/√d_head (the default), Hopfield retrieval must recover
    standard scaled dot-product attention: softmax(QK^T/√d) V.

    This is the testable consequence of Ramsauer et al.'s main theorem.
    """

    def test_matches_scaled_dot_product_no_mask(self) -> None:
        attn = HopfieldAttention(d_head=D_HEAD)
        q, k, v = _make_qkv()

        hopfield_out = attn(q, k, v)

        # Reference: standard scaled dot-product attention
        scale = 1.0 / (D_HEAD ** 0.5)
        scores = q @ k.transpose(-2, -1) * scale
        ref_out = F.softmax(scores, dim=-1) @ v

        torch.testing.assert_close(hopfield_out, ref_out, atol=1e-5, rtol=1e-5)

    def test_matches_scaled_dot_product_with_causal_mask(self) -> None:
        attn = HopfieldAttention(d_head=D_HEAD)
        q, k, v = _make_qkv()
        mask = _causal_mask(SEQ)

        hopfield_out = attn(q, k, v, causal_mask=mask)

        # Reference with causal masking
        scale = 1.0 / (D_HEAD ** 0.5)
        scores = q @ k.transpose(-2, -1) * scale
        scores = scores.masked_fill(mask, float("-inf"))
        ref_out = F.softmax(scores, dim=-1) @ v

        torch.testing.assert_close(hopfield_out, ref_out, atol=1e-5, rtol=1e-5)

    def test_explicit_beta_matches_default(self) -> None:
        """Explicitly passing β = 1/√d yields same result as default (None)."""
        default_attn = HopfieldAttention(d_head=D_HEAD)
        explicit_attn = HopfieldAttention(
            d_head=D_HEAD,
            config=HopfieldConfig(beta=1.0 / (D_HEAD ** 0.5)),
        )
        q, k, v = _make_qkv()

        out_default = default_attn(q, k, v)
        out_explicit = explicit_attn(q, k, v)

        torch.testing.assert_close(out_default, out_explicit, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Causal masking tests
# ---------------------------------------------------------------------------


class TestCausalMasking:
    """Verify that causal mask prevents future information leakage."""

    def test_first_token_attends_only_to_self(self) -> None:
        """Position 0 should attend only to position 0 (all later tokens masked)."""
        q, k, v = _make_qkv()
        mask = _causal_mask(SEQ)
        beta = torch.tensor(1.0 / (D_HEAD ** 0.5))

        # Manually compute attention weights to inspect them
        scores = beta * (q @ k.transpose(-2, -1))
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        # First token: all weight on position 0
        first_token_weights = weights[:, :, 0, :]  # (B, H, M)
        assert torch.allclose(
            first_token_weights[:, :, 0],
            torch.ones(BATCH, HEADS),
            atol=1e-6,
        )
        assert torch.allclose(
            first_token_weights[:, :, 1:],
            torch.zeros(BATCH, HEADS, SEQ - 1),
            atol=1e-6,
        )

    def test_causal_changes_output(self) -> None:
        """Masked output should differ from unmasked (unless seq_len=1)."""
        attn = HopfieldAttention(d_head=D_HEAD)
        q, k, v = _make_qkv()
        mask = _causal_mask(SEQ)

        out_no_mask = attn(q, k, v)
        out_masked = attn(q, k, v, causal_mask=mask)

        # They should differ for seq_len > 1
        assert not torch.allclose(out_no_mask, out_masked, atol=1e-5)


# ---------------------------------------------------------------------------
# Beta (inverse temperature) behavior
# ---------------------------------------------------------------------------


class TestBetaBehavior:
    """Verify β controls attention sharpness as predicted by Hopfield theory."""

    def test_high_beta_sharpens_attention(self) -> None:
        """High β should produce near-one-hot attention (winner-take-all)."""
        q, k, v = _make_qkv(seq_len=4)
        beta_high = torch.tensor(100.0)

        scores = beta_high * (q @ k.transpose(-2, -1))
        weights = F.softmax(scores, dim=-1)

        # Each row should be close to one-hot — max weight near 1.0
        max_weights = weights.max(dim=-1).values
        assert (max_weights > 0.99).all(), (
            f"High β should give near-one-hot attention, "
            f"but min max-weight was {max_weights.min().item():.4f}"
        )

    def test_low_beta_softens_attention(self) -> None:
        """Low β should produce near-uniform attention."""
        q, k, v = _make_qkv(seq_len=4)
        beta_low = torch.tensor(0.001)

        scores = beta_low * (q @ k.transpose(-2, -1))
        weights = F.softmax(scores, dim=-1)

        # Each row should be close to uniform = 1/seq_len = 0.25
        expected_uniform = 1.0 / 4
        assert torch.allclose(
            weights, torch.full_like(weights, expected_uniform), atol=0.01
        )

    def test_different_beta_different_output(self) -> None:
        """Two different β values must produce different outputs."""
        attn_low = HopfieldAttention(d_head=D_HEAD, config=HopfieldConfig(beta=0.01))
        attn_high = HopfieldAttention(d_head=D_HEAD, config=HopfieldConfig(beta=1.0))
        q, k, v = _make_qkv()

        out_low = attn_low(q, k, v)
        out_high = attn_high(q, k, v)

        assert not torch.allclose(out_low, out_high, atol=1e-5)


# ---------------------------------------------------------------------------
# Multi-iteration convergence
# ---------------------------------------------------------------------------


class TestMultiIteration:
    """Verify multi-step Hopfield retrieval converges toward a fixed point."""

    def test_single_vs_multi_differ(self) -> None:
        """Multiple iterations should change the output (not a no-op)."""
        attn_1 = HopfieldAttention(d_head=D_HEAD, config=HopfieldConfig(num_iters=1))
        attn_3 = HopfieldAttention(d_head=D_HEAD, config=HopfieldConfig(num_iters=3))
        q, k, v = _make_qkv()

        out_1 = attn_1(q, k, v)
        out_3 = attn_3(q, k, v)

        assert not torch.allclose(out_1, out_3, atol=1e-5)

    def test_convergence(self) -> None:
        """Successive iterations should converge: ||iter_n - iter_{n-1}|| decreases."""
        q, k, v = _make_qkv()
        beta = torch.tensor(1.0 / (D_HEAD ** 0.5))

        prev = q
        prev_delta = float("inf")
        for _ in range(5):
            curr = hopfield_retrieval(prev, k, v, beta)
            delta = (curr - prev).norm().item()
            assert delta < prev_delta or delta < 1e-4, (
                f"Retrieval should converge: delta={delta:.6f} >= prev_delta={prev_delta:.6f}"
            )
            prev_delta = delta
            prev = curr


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Verify no NaN or inf even under extreme conditions."""

    def test_large_beta_no_nan(self) -> None:
        """Very large β must not produce NaN (F.softmax handles this)."""
        attn = HopfieldAttention(d_head=D_HEAD, config=HopfieldConfig(beta=1000.0))
        q, k, v = _make_qkv()
        out = attn(q, k, v)
        assert not torch.isnan(out).any(), "NaN in output with large beta"
        assert not torch.isinf(out).any(), "Inf in output with large beta"

    def test_small_beta_no_nan(self) -> None:
        """Very small β must not produce NaN."""
        attn = HopfieldAttention(d_head=D_HEAD, config=HopfieldConfig(beta=1e-6))
        q, k, v = _make_qkv()
        out = attn(q, k, v)
        assert not torch.isnan(out).any(), "NaN in output with small beta"
        assert not torch.isinf(out).any(), "Inf in output with small beta"

    def test_large_values_no_overflow(self) -> None:
        """Large input magnitudes should not cause overflow."""
        attn = HopfieldAttention(d_head=D_HEAD)
        torch.manual_seed(42)
        q = torch.randn(1, 1, 4, D_HEAD) * 100
        k = torch.randn(1, 1, 4, D_HEAD) * 100
        v = torch.randn(1, 1, 4, D_HEAD)
        out = attn(q, k, v)
        assert not torch.isnan(out).any(), "NaN with large input magnitude"
        assert not torch.isinf(out).any(), "Inf with large input magnitude"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Gradients must propagate through Hopfield retrieval to Q, K, V."""

    def test_gradients_to_qkv(self) -> None:
        attn = HopfieldAttention(d_head=D_HEAD)
        q, k, v = _make_qkv()
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        loss = attn(q, k, v).sum()
        loss.backward()

        assert q.grad is not None, "No gradient on queries"
        assert k.grad is not None, "No gradient on keys"
        assert v.grad is not None, "No gradient on values"

    def test_gradients_with_causal_mask(self) -> None:
        """Causal mask should not block gradient flow."""
        attn = HopfieldAttention(d_head=D_HEAD)
        q, k, v = _make_qkv()
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        mask = _causal_mask(SEQ)

        loss = attn(q, k, v, causal_mask=mask).sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

    def test_gradients_multi_iter(self) -> None:
        """Multi-iteration retrieval must still propagate gradients."""
        attn = HopfieldAttention(d_head=D_HEAD, config=HopfieldConfig(num_iters=3))
        q, k, v = _make_qkv()
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        loss = attn(q, k, v).sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None


# ---------------------------------------------------------------------------
# Buffer behavior (device portability)
# ---------------------------------------------------------------------------


class TestBufferBehavior:
    """Verify beta is a buffer (moves with .to(), not a parameter)."""

    def test_beta_is_buffer_not_parameter(self) -> None:
        attn = HopfieldAttention(d_head=D_HEAD)
        param_names = {name for name, _ in attn.named_parameters()}
        buffer_names = {name for name, _ in attn.named_buffers()}

        assert "beta" not in param_names, "beta should not be a learnable parameter"
        assert "beta" in buffer_names, "beta should be a registered buffer"

    def test_no_learnable_parameters(self) -> None:
        """HopfieldAttention has no learnable parameters — it's a fixed mechanism."""
        attn = HopfieldAttention(d_head=D_HEAD)
        trainable = sum(p.numel() for p in attn.parameters() if p.requires_grad)
        assert trainable == 0, (
            f"HopfieldAttention should have 0 trainable params, got {trainable}"
        )

    def test_default_beta_value(self) -> None:
        """Default β should equal 1/√d_head."""
        attn = HopfieldAttention(d_head=D_HEAD)
        expected = 1.0 / (D_HEAD ** 0.5)
        assert torch.isclose(attn.beta, torch.tensor(expected), atol=1e-7)


# ---------------------------------------------------------------------------
# extra_repr
# ---------------------------------------------------------------------------


class TestRepr:
    """Verify informative repr for debugging."""

    def test_extra_repr_contains_config(self) -> None:
        attn = HopfieldAttention(d_head=64, config=HopfieldConfig(beta=0.5, num_iters=2))
        repr_str = repr(attn)
        assert "d_head=64" in repr_str
        assert "beta=0.5" in repr_str
        assert "num_iters=2" in repr_str
