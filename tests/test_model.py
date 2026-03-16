"""Tests for GPT-2 model loading, Conv1D conversion, and LoRA injection.

Tests are ordered by dependency: Conv1D conversion -> model loading ->
injection -> verification. All tests run on CPU (no GPU required).
GPT-2 is loaded once per test class via fixtures to avoid redundant downloads.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import GPT2Model
from transformers.pytorch_utils import Conv1D

from src.lora import LoRAConfig, LoRALinear
from src.model import (
    conv1d_to_linear,
    freeze_all_parameters,
    inject_lora,
    load_gpt2,
    unfreeze_all_parameters,
    verify_lora_injection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpt2_model() -> GPT2Model:
    """Load GPT-2 small once for the entire test module."""
    return load_gpt2("gpt2")


# ---------------------------------------------------------------------------
# Conv1D -> nn.Linear conversion
# ---------------------------------------------------------------------------


class TestConv1dToLinear:
    """Verify that Conv1D conversion produces numerically identical results."""

    def test_output_equivalence_square(self) -> None:
        """Square weight matrix: (768, 768) like c_proj."""
        conv = Conv1D(nf=768, nx=768)
        linear = conv1d_to_linear(conv)

        x = torch.randn(2, 10, 768)
        torch.testing.assert_close(conv(x), linear(x), atol=1e-5, rtol=1e-5)

    def test_output_equivalence_rectangular(self) -> None:
        """Rectangular weight matrix: (768, 2304) like c_attn."""
        conv = Conv1D(nf=2304, nx=768)
        linear = conv1d_to_linear(conv)

        x = torch.randn(2, 10, 768)
        torch.testing.assert_close(conv(x), linear(x), atol=1e-5, rtol=1e-5)

    def test_weight_shapes(self) -> None:
        """nn.Linear weight should be transposed relative to Conv1D."""
        conv = Conv1D(nf=2304, nx=768)
        linear = conv1d_to_linear(conv)

        assert conv.weight.shape == (768, 2304)
        assert linear.weight.shape == (2304, 768)

    def test_does_not_share_memory(self) -> None:
        """Modifying the converted linear must not affect the original Conv1D."""
        conv = Conv1D(nf=32, nx=16)
        linear = conv1d_to_linear(conv)

        original_conv_weight = conv.weight.data.clone()
        linear.weight.data.zero_()

        assert torch.equal(conv.weight.data, original_conv_weight)

    def test_bias_copied(self) -> None:
        """Bias should be copied, not shared."""
        conv = Conv1D(nf=32, nx=16)
        linear = conv1d_to_linear(conv)

        torch.testing.assert_close(conv.bias.data, linear.bias.data)

        # Verify independence
        original_bias = conv.bias.data.clone()
        linear.bias.data.zero_()
        assert torch.equal(conv.bias.data, original_bias)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


class TestLoadGpt2:
    """Verify GPT2Model loading and output structure."""

    def test_returns_gpt2model(self, gpt2_model: GPT2Model) -> None:
        assert isinstance(gpt2_model, GPT2Model)

    def test_has_last_hidden_state(self, gpt2_model: GPT2Model) -> None:
        """train.py accesses outputs.last_hidden_state — must exist."""
        x = torch.randint(0, 1000, (1, 5))
        outputs = gpt2_model(x)
        assert hasattr(outputs, "last_hidden_state")
        assert outputs.last_hidden_state.shape == (1, 5, 768)

    def test_num_blocks(self, gpt2_model: GPT2Model) -> None:
        assert len(gpt2_model.h) == 12

    def test_hidden_size(self, gpt2_model: GPT2Model) -> None:
        assert gpt2_model.config.hidden_size == 768

    def test_attention_layers_are_conv1d(self, gpt2_model: GPT2Model) -> None:
        """Before injection, attention projections should be Conv1D."""
        block = gpt2_model.h[0]
        assert isinstance(block.attn.c_attn, Conv1D)
        assert isinstance(block.attn.c_proj, Conv1D)


# ---------------------------------------------------------------------------
# Freeze/unfreeze
# ---------------------------------------------------------------------------


class TestFreezeUnfreeze:
    """Verify parameter freezing and unfreezing."""

    def test_freeze_all(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        freeze_all_parameters(model)
        for p in model.parameters():
            assert not p.requires_grad

    def test_unfreeze_all(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        freeze_all_parameters(model)
        unfreeze_all_parameters(model)
        for p in model.parameters():
            assert p.requires_grad


# ---------------------------------------------------------------------------
# LoRA injection
# ---------------------------------------------------------------------------


class TestInjectLora:
    """Verify LoRA injection into GPT-2's Conv1D attention layers."""

    def test_replaces_c_attn(self) -> None:
        model = load_gpt2()
        config = LoRAConfig(r=4, alpha=4.0, target_modules=("c_attn",))
        replaced = inject_lora(model, config)

        # Should replace c_attn in all 12 blocks
        assert len(replaced) == 12
        for block_idx in range(12):
            path = f"h.{block_idx}.attn.c_attn"
            assert path in replaced
            assert isinstance(model.h[block_idx].attn.c_attn, LoRALinear)

    def test_replaces_c_attn_and_c_proj(self) -> None:
        model = load_gpt2()
        config = LoRAConfig(r=4, alpha=4.0, target_modules=("c_attn", "c_proj"))
        replaced = inject_lora(model, config)

        # 12 blocks * 2 targets = 24 replacements
        assert len(replaced) == 24

    def test_c_proj_not_replaced_when_not_targeted(self) -> None:
        model = load_gpt2()
        config = LoRAConfig(r=4, alpha=4.0, target_modules=("c_attn",))
        inject_lora(model, config)

        # c_proj should still be Conv1D
        assert isinstance(model.h[0].attn.c_proj, Conv1D)

    def test_trainable_param_count_c_attn(self) -> None:
        """LoRA r=4 on c_attn (768->2304): r*(in+out) = 4*(768+2304) = 12288 per block."""
        model = load_gpt2()
        config = LoRAConfig(r=4, alpha=4.0, target_modules=("c_attn",))
        replaced = inject_lora(model, config)

        expected_per_block = 4 * (768 + 2304)  # 12,288
        for path, count in replaced.items():
            assert count == expected_per_block, f"{path}: expected {expected_per_block}, got {count}"

        # Total trainable in model (adapter params only)
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_trainable == 12 * expected_per_block  # 147,456

    def test_trainable_param_count_various_ranks(self) -> None:
        """Verify param count scales linearly with rank."""
        for r in [1, 4, 8, 16]:
            model = load_gpt2()
            config = LoRAConfig(r=r, alpha=float(r), target_modules=("c_attn",))
            inject_lora(model, config)

            expected_total = 12 * r * (768 + 2304)
            actual_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert actual_total == expected_total, f"r={r}: expected {expected_total}, got {actual_total}"

    def test_base_weights_frozen_after_injection(self) -> None:
        model = load_gpt2()
        config = LoRAConfig(r=4, alpha=4.0, target_modules=("c_attn",))
        inject_lora(model, config)

        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert param.requires_grad, f"Adapter {name} should be trainable"
            else:
                assert not param.requires_grad, f"Base param {name} should be frozen"

    def test_invalid_target_raises(self) -> None:
        model = load_gpt2()
        config = LoRAConfig(r=4, target_modules=("nonexistent_module",))
        with pytest.raises(ValueError, match="no attribute"):
            inject_lora(model, config)

    def test_output_shape_preserved(self) -> None:
        """LoRA-injected model must produce same output shape as original."""
        model = load_gpt2()
        config = LoRAConfig(r=4, alpha=4.0, target_modules=("c_attn",))
        inject_lora(model, config)

        x = torch.randint(0, 1000, (2, 10))
        outputs = model(x)
        assert outputs.last_hidden_state.shape == (2, 10, 768)


# ---------------------------------------------------------------------------
# Output equivalence at initialization
# ---------------------------------------------------------------------------


class TestOutputEquivalenceAtInit:
    """Because B is zero-init, a freshly injected LoRA model should produce
    output identical to the original (non-injected) model."""

    def test_identical_output_at_init(self) -> None:
        torch.manual_seed(42)
        x = torch.randint(0, 1000, (2, 10))

        # Get reference output from vanilla model
        model_ref = load_gpt2()
        model_ref.eval()
        with torch.no_grad():
            ref_out = model_ref(x).last_hidden_state

        # Inject LoRA and compare
        model_lora = load_gpt2()
        config = LoRAConfig(r=4, alpha=4.0, target_modules=("c_attn",))
        inject_lora(model_lora, config)
        model_lora.eval()
        with torch.no_grad():
            lora_out = model_lora(x).last_hidden_state

        torch.testing.assert_close(ref_out, lora_out, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Verification function
# ---------------------------------------------------------------------------


class TestVerifyLoraInjection:
    """Verify the diagnostic function catches correct and incorrect setups."""

    def test_passes_on_correct_injection(self) -> None:
        model = load_gpt2()
        config = LoRAConfig(r=4, alpha=4.0, target_modules=("c_attn",))
        inject_lora(model, config)

        summary = verify_lora_injection(model, config)
        assert summary["all_checks_passed"] is True
        assert len(summary["lora_modules"]) == 12
        assert summary["trainable_params"] == 12 * 4 * (768 + 2304)

    def test_passes_with_c_attn_and_c_proj(self) -> None:
        model = load_gpt2()
        config = LoRAConfig(r=8, alpha=8.0, target_modules=("c_attn", "c_proj"))
        inject_lora(model, config)

        summary = verify_lora_injection(model, config)
        assert summary["all_checks_passed"] is True
        assert len(summary["lora_modules"]) == 24

        expected = 12 * 8 * (768 + 2304) + 12 * 8 * (768 + 768)
        assert summary["trainable_params"] == expected


# ---------------------------------------------------------------------------
# Frozen baseline and full fine-tune configurations
# ---------------------------------------------------------------------------


class TestFrozenBaseline:
    """Frozen baseline: all params frozen, zero trainable."""

    def test_zero_trainable_params(self) -> None:
        model = load_gpt2()
        freeze_all_parameters(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == 0

    def test_still_produces_output(self) -> None:
        model = load_gpt2()
        freeze_all_parameters(model)
        model.eval()
        x = torch.randint(0, 1000, (1, 5))
        with torch.no_grad():
            out = model(x)
        assert out.last_hidden_state.shape == (1, 5, 768)


class TestFullFineTune:
    """Full fine-tune: all params trainable."""

    def test_all_params_trainable(self) -> None:
        model = load_gpt2()
        unfreeze_all_parameters(model)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == total
