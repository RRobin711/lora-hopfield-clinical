"""Tests for Hopfield attention injection into GPT-2.

Core correctness gate: at beta = 1/sqrt(d_head) = 1/sqrt(64), the Hopfield-injected
model must produce numerically identical output to vanilla GPT-2. This is the
theoretical guarantee from Ramsauer et al. (2020) — one Hopfield retrieval step
with inverse temperature 1/sqrt(d) IS standard scaled dot-product attention.

Tests are ordered by dependency: injection mechanics -> output equivalence ->
verification -> training compatibility.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import GPT2Model

from src.hopfield import HopfieldAttention, HopfieldConfig
from src.hopfield_gpt2 import (
    inject_hopfield,
    verify_hopfield_injection,
)
from src.model import load_gpt2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpt2_model() -> GPT2Model:
    """Load GPT-2 small once for the entire test module."""
    return load_gpt2("gpt2")


# ---------------------------------------------------------------------------
# Injection mechanics
# ---------------------------------------------------------------------------


class TestInjectHopfield:
    """Verify that inject_hopfield correctly modifies the model structure."""

    def test_all_blocks_have_hopfield(self) -> None:
        model = load_gpt2()
        inject_hopfield(model)

        for block_idx in range(12):
            attn = model.h[block_idx].attn
            assert hasattr(attn, "hopfield_attn"), (
                f"h.{block_idx}.attn missing hopfield_attn"
            )
            assert isinstance(attn.hopfield_attn, HopfieldAttention)

    def test_all_base_params_frozen(self) -> None:
        model = load_gpt2()
        inject_hopfield(model)

        for name, param in model.named_parameters():
            assert not param.requires_grad, (
                f"Parameter should be frozen: {name}"
            )

    def test_returns_12_blocks(self) -> None:
        model = load_gpt2()
        replaced = inject_hopfield(model)
        assert len(replaced) == 12

    def test_zero_trainable_params(self) -> None:
        """Hopfield adds no trainable params — only a beta buffer per block."""
        model = load_gpt2()
        inject_hopfield(model)
        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        assert trainable == 0

    def test_beta_buffer_on_each_block(self) -> None:
        """Each HopfieldAttention should have a beta buffer (not a parameter)."""
        model = load_gpt2()
        config = HopfieldConfig(beta=0.25)
        inject_hopfield(model, config)

        for block_idx in range(12):
            hop = model.h[block_idx].attn.hopfield_attn
            assert hasattr(hop, "beta")
            assert torch.isclose(hop.beta, torch.tensor(0.25))
            # Beta should be a buffer, not a parameter
            param_names = {n for n, _ in hop.named_parameters()}
            assert "beta" not in param_names

    def test_custom_config_propagated(self) -> None:
        model = load_gpt2()
        config = HopfieldConfig(beta=0.5, num_iters=1, dropout=0.1)
        inject_hopfield(model, config)

        hop = model.h[0].attn.hopfield_attn
        assert torch.isclose(hop.beta, torch.tensor(0.5))
        assert hop.num_iters == 1
        assert hop.dropout_p == 0.1


# ---------------------------------------------------------------------------
# Output equivalence — the correctness gate
# ---------------------------------------------------------------------------


class TestBetaRecovery:
    """At beta = 1/sqrt(d_head), Hopfield must recover standard attention.

    This is the fundamental theorem from Ramsauer et al.: the Hopfield retrieval
    rule with inverse temperature 1/sqrt(d) is algebraically identical to
    scaled dot-product attention. If this test fails, the injection is broken.
    """

    def test_identical_output_default_beta(self) -> None:
        """Default beta (1/sqrt(64) = 0.125) must give identical output."""
        torch.manual_seed(42)
        x = torch.randint(0, 1000, (2, 10))

        # Reference: vanilla GPT-2
        model_ref = load_gpt2()
        model_ref.eval()
        with torch.no_grad():
            ref_out = model_ref(x).last_hidden_state

        # Hopfield with default config (beta = 1/sqrt(64))
        model_hop = load_gpt2()
        inject_hopfield(model_hop)
        model_hop.eval()
        with torch.no_grad():
            hop_out = model_hop(x).last_hidden_state

        torch.testing.assert_close(ref_out, hop_out, atol=1e-4, rtol=1e-4)

    def test_identical_output_with_attention_mask(self) -> None:
        """Recovery must hold when padding mask is applied."""
        torch.manual_seed(42)
        x = torch.randint(0, 1000, (2, 10))
        # Simulate padding: second sequence has 7 real tokens, 3 pads
        attention_mask = torch.ones(2, 10, dtype=torch.long)
        attention_mask[1, 7:] = 0

        model_ref = load_gpt2()
        model_ref.eval()
        with torch.no_grad():
            ref_out = model_ref(x, attention_mask=attention_mask).last_hidden_state

        model_hop = load_gpt2()
        inject_hopfield(model_hop)
        model_hop.eval()
        with torch.no_grad():
            hop_out = model_hop(x, attention_mask=attention_mask).last_hidden_state

        torch.testing.assert_close(ref_out, hop_out, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Output shape preservation
# ---------------------------------------------------------------------------


class TestOutputShape:
    """Verify Hopfield-injected model produces correct output shapes."""

    def test_last_hidden_state_shape(self) -> None:
        model = load_gpt2()
        inject_hopfield(model)
        model.eval()

        x = torch.randint(0, 1000, (2, 10))
        with torch.no_grad():
            outputs = model(x)

        assert hasattr(outputs, "last_hidden_state")
        assert outputs.last_hidden_state.shape == (2, 10, 768)

    def test_variable_sequence_length(self) -> None:
        model = load_gpt2()
        inject_hopfield(model)
        model.eval()

        for seq_len in [5, 32, 128]:
            x = torch.randint(0, 1000, (1, seq_len))
            with torch.no_grad():
                out = model(x).last_hidden_state
            assert out.shape == (1, seq_len, 768)

    def test_single_token_sequence(self) -> None:
        """Edge case: single token should still produce valid output."""
        model = load_gpt2()
        inject_hopfield(model)
        model.eval()

        x = torch.randint(0, 1000, (1, 1))
        with torch.no_grad():
            out = model(x).last_hidden_state
        assert out.shape == (1, 1, 768)


# ---------------------------------------------------------------------------
# Non-default beta changes output
# ---------------------------------------------------------------------------


class TestBetaEffect:
    """Verify that changing beta actually changes the attention pattern."""

    def test_different_beta_gives_different_output(self) -> None:
        """Non-default beta should produce different (but valid) output."""
        torch.manual_seed(42)
        x = torch.randint(0, 1000, (2, 10))

        model_default = load_gpt2()
        inject_hopfield(model_default, HopfieldConfig(beta=None))  # 1/sqrt(64)
        model_default.eval()

        model_sharp = load_gpt2()
        inject_hopfield(model_sharp, HopfieldConfig(beta=1.0))  # 8x sharper
        model_sharp.eval()

        with torch.no_grad():
            out_default = model_default(x).last_hidden_state
            out_sharp = model_sharp(x).last_hidden_state

        # Outputs should differ — higher beta sharpens attention
        assert not torch.allclose(out_default, out_sharp, atol=1e-3)
        # But both should have the right shape and be finite
        assert out_sharp.shape == (2, 10, 768)
        assert torch.isfinite(out_sharp).all()


# ---------------------------------------------------------------------------
# Verification function
# ---------------------------------------------------------------------------


class TestVerifyHopfieldInjection:
    """Verify the diagnostic function works correctly."""

    def test_passes_on_correct_injection(self) -> None:
        model = load_gpt2()
        inject_hopfield(model)

        summary = verify_hopfield_injection(model)
        assert summary["all_checks_passed"] is True
        assert summary["blocks_with_hopfield"] == 12
        assert summary["trainable_params"] == 0

    def test_fails_on_uninjected_model(self) -> None:
        model = load_gpt2()
        summary = verify_hopfield_injection(model)
        assert summary["all_checks_passed"] is False
        assert summary["blocks_with_hopfield"] == 0


# ---------------------------------------------------------------------------
# Gradient flow for training
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Verify gradients flow correctly through the Hopfield-injected model.

    All GPT-2 base params are frozen. Only an external classification head
    should accumulate gradients. The backward pass must flow through the
    Hopfield attention computation without error.
    """

    def test_backward_pass_succeeds(self) -> None:
        model = load_gpt2()
        inject_hopfield(model)

        head = nn.Linear(768, 2)
        x = torch.randint(0, 1000, (2, 10))
        mask = torch.ones(2, 10, dtype=torch.long)

        outputs = model(x, attention_mask=mask)
        # Extract last token hidden state (simplified — no padding here)
        pooled = outputs.last_hidden_state[:, -1, :]
        logits = head(pooled)
        loss = logits.sum()
        loss.backward()

        # Head should have gradients
        assert head.weight.grad is not None
        assert head.bias.grad is not None

        # All model params should have NO gradients (frozen)
        for name, param in model.named_parameters():
            assert param.grad is None or torch.all(param.grad == 0), (
                f"Frozen param {name} has unexpected gradient"
            )

    def test_head_params_trainable_model_params_frozen(self) -> None:
        """Simulate the train.py optimizer setup."""
        model = load_gpt2()
        inject_hopfield(model)
        head = nn.Linear(768, 2)

        trainable = [p for p in model.parameters() if p.requires_grad]
        trainable += list(head.parameters())

        # Only head params should be trainable
        assert len(trainable) == 2  # head.weight + head.bias
        assert trainable[0].shape == (2, 768)  # weight
        assert trainable[1].shape == (2,)  # bias


# ---------------------------------------------------------------------------
# Device movement
# ---------------------------------------------------------------------------


class TestDeviceMovement:
    """Verify HopfieldAttention buffers move with .to(device)."""

    def test_beta_moves_to_device(self) -> None:
        model = load_gpt2()
        inject_hopfield(model)

        # Check beta is on CPU initially
        hop = model.h[0].attn.hopfield_attn
        assert hop.beta.device.type == "cpu"

        # Moving model should move beta too (since it's a registered buffer)
        model = model.to("cpu")  # no-op but validates the path
        assert hop.beta.device.type == "cpu"
