"""Tests for LoRALinear module.

Verifies the core invariants from Hu et al. (2021):
- Zero-init: adapter contributes nothing at initialization
- Gradient flow: only adapter parameters receive gradients
- Shape preservation: output matches standard nn.Linear
- Parameter count: exactly r * (in_features + out_features) trainable params
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.lora import LoRAConfig, LoRALinear


class TestLoRAConfig:
    """Verify config dataclass defaults and structure."""

    def test_defaults(self) -> None:
        cfg = LoRAConfig()
        assert cfg.r == 4
        assert cfg.alpha == 4.0
        assert cfg.dropout == 0.0
        assert cfg.target_modules == ("c_attn",)

    def test_custom_values(self) -> None:
        cfg = LoRAConfig(r=16, alpha=32.0, dropout=0.1, target_modules=("c_attn", "c_proj"))
        assert cfg.r == 16
        assert cfg.alpha == 32.0
        assert cfg.target_modules == ("c_attn", "c_proj")


class TestLoRALinearInit:
    """Verify initialization invariants."""

    def test_zero_init_B(self) -> None:
        """B must be zero so the adapter is identity at t=0 (Hu et al. Section 4)."""
        layer = LoRALinear(64, 128, r=4, alpha=4.0)
        assert torch.allclose(layer.lora_B, torch.zeros_like(layer.lora_B))

    def test_A_is_not_zero(self) -> None:
        """A should be Kaiming-initialized, not zero — otherwise BA is always zero."""
        layer = LoRALinear(64, 128, r=4, alpha=4.0)
        assert not torch.allclose(layer.lora_A, torch.zeros_like(layer.lora_A))

    def test_base_weight_frozen(self) -> None:
        """Frozen base weight must not have requires_grad."""
        layer = LoRALinear(64, 128, r=4, alpha=4.0)
        assert not layer.linear.weight.requires_grad

    def test_base_bias_frozen(self) -> None:
        """Frozen base bias must not have requires_grad."""
        layer = LoRALinear(64, 128, r=4, alpha=4.0, bias=True)
        assert layer.linear.bias is not None
        assert not layer.linear.bias.requires_grad

    def test_no_bias(self) -> None:
        """Layer without bias should not create a bias parameter."""
        layer = LoRALinear(64, 128, r=4, alpha=4.0, bias=False)
        assert layer.linear.bias is None

    def test_adapter_shapes(self) -> None:
        """A is (r, in_features), B is (out_features, r)."""
        layer = LoRALinear(64, 128, r=8, alpha=8.0)
        assert layer.lora_A.shape == (8, 64)
        assert layer.lora_B.shape == (128, 8)

    def test_scaling_factor(self) -> None:
        """Scaling = alpha / r."""
        layer = LoRALinear(64, 128, r=4, alpha=16.0)
        assert layer.scaling == pytest.approx(4.0)


class TestLoRALinearValidation:
    """Verify input validation catches bad configurations."""

    def test_rank_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="rank r must be positive"):
            LoRALinear(64, 128, r=0)

    def test_rank_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="rank r must be positive"):
            LoRALinear(64, 128, r=-1)

    def test_rank_exceeds_dimensions_raises(self) -> None:
        with pytest.raises(ValueError, match="would not be low-rank"):
            LoRALinear(8, 128, r=16)


class TestLoRALinearForward:
    """Verify forward pass behavior."""

    def test_output_shape_2d(self) -> None:
        """Simple (batch, features) input."""
        layer = LoRALinear(64, 128, r=4, alpha=4.0)
        x = torch.randn(8, 64)
        out = layer(x)
        assert out.shape == (8, 128)

    def test_output_shape_3d(self) -> None:
        """(batch, seq_len, features) — typical transformer input."""
        layer = LoRALinear(768, 768, r=4, alpha=4.0)
        x = torch.randn(2, 10, 768)
        out = layer(x)
        assert out.shape == (2, 10, 768)

    def test_output_shape_4d(self) -> None:
        """Arbitrary leading dims should pass through."""
        layer = LoRALinear(32, 64, r=4, alpha=4.0)
        x = torch.randn(2, 3, 5, 32)
        out = layer(x)
        assert out.shape == (2, 3, 5, 64)

    def test_identity_at_init(self) -> None:
        """At t=0, adapter output is zero so total output equals base linear output.

        This verifies the zero-init invariant end-to-end: the LoRALinear forward
        pass must produce identical results to the frozen base nn.Linear at init.
        """
        layer = LoRALinear(64, 128, r=4, alpha=4.0)
        x = torch.randn(4, 10, 64)

        lora_out = layer(x)
        base_out = layer.linear(x)

        assert torch.allclose(lora_out, base_out, atol=1e-6)

    def test_adapter_contributes_after_update(self) -> None:
        """After modifying B away from zero, output should differ from base."""
        layer = LoRALinear(64, 128, r=4, alpha=4.0)
        x = torch.randn(4, 64)

        # Manually set B to nonzero to simulate a training step
        layer.lora_B.data.fill_(0.1)

        lora_out = layer(x)
        base_out = layer.linear(x)

        assert not torch.allclose(lora_out, base_out, atol=1e-6)


class TestLoRALinearGradients:
    """Verify gradient flow — the most critical correctness test.

    LoRA's value proposition depends on gradients flowing ONLY to the adapter
    parameters (A and B) while the base weight remains frozen. If gradients
    leak to the base weight, we're doing full fine-tuning with extra overhead.
    """

    def test_gradients_flow_to_adapter(self) -> None:
        """A and B must receive gradients after backward pass."""
        layer = LoRALinear(64, 64, r=4, alpha=4.0)
        x = torch.randn(2, 64)
        loss = layer(x).sum()
        loss.backward()

        assert layer.lora_A.grad is not None, "lora_A should have gradients"
        assert layer.lora_B.grad is not None, "lora_B should have gradients"

    def test_no_gradients_to_frozen_weight(self) -> None:
        """Base weight must not accumulate gradients."""
        layer = LoRALinear(64, 64, r=4, alpha=4.0)
        x = torch.randn(2, 64)
        loss = layer(x).sum()
        loss.backward()

        assert layer.linear.weight.grad is None, "Frozen base weight should not have gradients"

    def test_no_gradients_to_frozen_bias(self) -> None:
        """Base bias must not accumulate gradients."""
        layer = LoRALinear(64, 64, r=4, alpha=4.0, bias=True)
        x = torch.randn(2, 64)
        loss = layer(x).sum()
        loss.backward()

        assert layer.linear.bias.grad is None, "Frozen base bias should not have gradients"

    def test_gradient_magnitude_scales_with_alpha(self) -> None:
        """Higher alpha should produce larger gradients through the adapter path.

        This isn't a precise numerical test — it verifies the scaling factor
        is actually wired into the computation graph, not just stored as metadata.
        """
        # First: small alpha
        layer_small = LoRALinear(64, 64, r=4, alpha=1.0)
        # Second: large alpha, same A init for fair comparison
        layer_large = LoRALinear(64, 64, r=4, alpha=16.0)
        layer_large.lora_A.data.copy_(layer_small.lora_A.data)

        # Use same input and nonzero B so gradients are nonzero
        x = torch.randn(2, 64)
        layer_small.lora_B.data.fill_(0.1)
        layer_large.lora_B.data.fill_(0.1)

        loss_small = layer_small(x).sum()
        loss_small.backward()
        grad_norm_small = layer_small.lora_A.grad.norm().item()

        loss_large = layer_large(x).sum()
        loss_large.backward()
        grad_norm_large = layer_large.lora_A.grad.norm().item()

        assert grad_norm_large > grad_norm_small, (
            f"Larger alpha should produce larger gradients: "
            f"alpha=16 grad_norm={grad_norm_large:.4f}, alpha=1 grad_norm={grad_norm_small:.4f}"
        )


class TestLoRALinearParameterCount:
    """Verify trainable parameter count is exactly r * (in_features + out_features).

    This catches common bugs:
    - Accidentally leaving base weight trainable (count too high)
    - Doubling the adapter params (off-by-one in shape)
    - Forgetting to freeze bias
    """

    def test_trainable_params_no_bias(self) -> None:
        layer = LoRALinear(64, 128, r=4, alpha=4.0, bias=False)
        trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        # A: (4, 64) = 256, B: (128, 4) = 512, total = 768
        expected = 4 * (64 + 128)
        assert trainable == expected, f"Expected {expected} trainable params, got {trainable}"

    def test_trainable_params_with_bias(self) -> None:
        """Bias is frozen — trainable count should be the same as without bias."""
        layer = LoRALinear(64, 128, r=4, alpha=4.0, bias=True)
        trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        expected = 4 * (64 + 128)
        assert trainable == expected, f"Expected {expected} trainable params, got {trainable}"

    def test_trainable_params_various_ranks(self) -> None:
        """Verify formula holds across the ablation ranks."""
        for r in [1, 4, 8, 16, 32]:
            layer = LoRALinear(768, 768, r=r, alpha=float(r))
            trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            expected = r * (768 + 768)
            assert trainable == expected, f"r={r}: expected {expected}, got {trainable}"


class TestLoRALinearFromLinear:
    """Verify the from_linear classmethod correctly wraps pretrained layers."""

    def test_weight_copy(self) -> None:
        """Pretrained weight must be faithfully copied into the frozen base."""
        original = nn.Linear(64, 128)
        lora_layer = LoRALinear.from_linear(original, r=4, alpha=4.0)

        assert torch.allclose(lora_layer.linear.weight.data, original.weight.data)

    def test_bias_copy(self) -> None:
        """Pretrained bias must be copied."""
        original = nn.Linear(64, 128, bias=True)
        lora_layer = LoRALinear.from_linear(original, r=4, alpha=4.0)

        assert torch.allclose(lora_layer.linear.bias.data, original.bias.data)

    def test_from_linear_preserves_output(self) -> None:
        """At t=0, from_linear output must match the original layer's output.

        This is the end-to-end check that from_linear + zero-init B produces
        a drop-in replacement for the original layer.
        """
        original = nn.Linear(64, 128)
        lora_layer = LoRALinear.from_linear(original, r=4, alpha=4.0)
        x = torch.randn(4, 10, 64)

        original_out = original(x)
        lora_out = lora_layer(x)

        assert torch.allclose(original_out, lora_out, atol=1e-6)

    def test_from_linear_no_bias(self) -> None:
        """from_linear respects the original layer's bias setting."""
        original = nn.Linear(64, 128, bias=False)
        lora_layer = LoRALinear.from_linear(original, r=4, alpha=4.0)

        assert lora_layer.linear.bias is None

    def test_from_linear_does_not_share_memory(self) -> None:
        """Weight should be copied, not aliased — modifying original shouldn't affect LoRA."""
        original = nn.Linear(64, 128)
        lora_layer = LoRALinear.from_linear(original, r=4, alpha=4.0)

        original.weight.data.fill_(999.0)
        assert not torch.allclose(lora_layer.linear.weight.data, original.weight.data)


class TestLoRALinearDropout:
    """Verify dropout is applied correctly in the adapter path."""

    def test_dropout_active_in_train_mode(self) -> None:
        """With dropout > 0 in train mode, outputs should have variance from dropout."""
        layer = LoRALinear(64, 64, r=4, alpha=4.0, dropout=0.5)
        layer.lora_B.data.fill_(0.1)  # Nonzero B so adapter path is active
        layer.train()

        x = torch.randn(32, 64)
        # Run forward twice — dropout should make outputs differ
        out1 = layer(x)
        out2 = layer(x)

        assert not torch.allclose(out1, out2, atol=1e-6), (
            "With dropout=0.5 in train mode, two forward passes should differ"
        )

    def test_dropout_inactive_in_eval_mode(self) -> None:
        """In eval mode, dropout is disabled — outputs should be deterministic."""
        layer = LoRALinear(64, 64, r=4, alpha=4.0, dropout=0.5)
        layer.lora_B.data.fill_(0.1)
        layer.eval()

        x = torch.randn(32, 64)
        out1 = layer(x)
        out2 = layer(x)

        assert torch.allclose(out1, out2, atol=1e-6), (
            "In eval mode, dropout should be disabled and outputs deterministic"
        )
