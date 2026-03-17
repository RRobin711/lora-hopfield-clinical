"""Tests for combined LoRA + Hopfield injection into GPT-2.

Verifies that injecting both LoRA adapters (learnable projections) and Hopfield
attention (energy-minimizing retrieval) into the same model works correctly when
done in the right order: Hopfield FIRST, then LoRA.

The key invariants:
1. Hopfield forward reads self.c_attn dynamically — LoRA's setattr replacement
   is picked up at call time without re-injecting the forward method.
2. LoRA adapter params (lora_A, lora_B) remain trainable after both injections.
3. All base GPT-2 params remain frozen.
4. Hopfield beta buffers persist through LoRA injection.
5. Forward produces finite output of correct shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.hopfield import HopfieldAttention, HopfieldConfig
from src.hopfield_gpt2 import inject_hopfield, verify_hopfield_injection
from src.lora import LoRAConfig, LoRALinear
from src.model import inject_lora, load_gpt2, verify_lora_injection


class TestCombinedInjection:
    """Verify combined Hopfield + LoRA injection produces a valid model."""

    def test_forward_produces_output(self) -> None:
        """Combined model should produce finite output of correct shape."""
        model = load_gpt2()
        inject_hopfield(model)
        inject_lora(model, LoRAConfig(r=4, alpha=4.0))
        model.eval()

        x = torch.randint(0, 1000, (2, 10))
        with torch.no_grad():
            out = model(x).last_hidden_state

        assert out.shape == (2, 10, 768)
        assert torch.isfinite(out).all()

    def test_lora_adapters_trainable(self) -> None:
        """LoRA adapter params must have requires_grad=True after combined injection."""
        model = load_gpt2()
        inject_hopfield(model)
        inject_lora(model, LoRAConfig(r=4, alpha=4.0))

        lora_params = [
            (name, p) for name, p in model.named_parameters()
            if "lora_A" in name or "lora_B" in name
        ]
        assert len(lora_params) > 0, "No LoRA params found"
        for name, param in lora_params:
            assert param.requires_grad, f"LoRA param {name} is not trainable"

    def test_base_params_frozen(self) -> None:
        """All non-LoRA params must be frozen after combined injection."""
        model = load_gpt2()
        inject_hopfield(model)
        inject_lora(model, LoRAConfig(r=4, alpha=4.0))

        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                continue
            assert not param.requires_grad, (
                f"Non-adapter param should be frozen: {name}"
            )

    def test_hopfield_buffers_present(self) -> None:
        """Hopfield beta buffers must survive LoRA injection."""
        model = load_gpt2()
        config = HopfieldConfig(beta=0.25)
        inject_hopfield(model, config)
        inject_lora(model, LoRAConfig(r=4, alpha=4.0))

        for block_idx in range(12):
            attn = model.h[block_idx].attn
            assert hasattr(attn, "hopfield_attn"), (
                f"h.{block_idx}.attn missing hopfield_attn after LoRA injection"
            )
            assert isinstance(attn.hopfield_attn, HopfieldAttention)
            assert torch.isclose(attn.hopfield_attn.beta, torch.tensor(0.25))

    def test_c_attn_is_lora_linear(self) -> None:
        """After combined injection, c_attn should be LoRALinear (not Conv1D)."""
        model = load_gpt2()
        inject_hopfield(model)
        inject_lora(model, LoRAConfig(r=4, alpha=4.0))

        for block_idx in range(12):
            c_attn = model.h[block_idx].attn.c_attn
            assert isinstance(c_attn, LoRALinear), (
                f"h.{block_idx}.attn.c_attn is {type(c_attn).__name__}, "
                "expected LoRALinear"
            )

    def test_verify_lora_passes(self) -> None:
        """verify_lora_injection() should pass after combined injection."""
        model = load_gpt2()
        inject_hopfield(model)
        lora_config = LoRAConfig(r=4, alpha=4.0)
        inject_lora(model, lora_config)

        summary = verify_lora_injection(model, lora_config)
        assert summary["all_checks_passed"] is True

    def test_trainable_param_count(self) -> None:
        """Trainable params = LoRA adapters only (same count as LoRA-only injection)."""
        model = load_gpt2()
        inject_hopfield(model)
        lora_config = LoRAConfig(r=4, alpha=4.0)
        inject_lora(model, lora_config)

        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        # r=4, c_attn: 4 * (768 + 2304) = 12,288 per block, 12 blocks = 147,456
        expected = 4 * (768 + 2304) * 12
        assert trainable == expected, (
            f"Expected {expected} trainable params, got {trainable}"
        )

    def test_backward_pass(self) -> None:
        """Backward pass should flow through combined Hopfield + LoRA."""
        model = load_gpt2()
        inject_hopfield(model)
        inject_lora(model, LoRAConfig(r=4, alpha=4.0))

        head = nn.Linear(768, 2)
        x = torch.randint(0, 1000, (2, 10))

        outputs = model(x)
        pooled = outputs.last_hidden_state[:, -1, :]
        logits = head(pooled)
        loss = logits.sum()
        loss.backward()

        # LoRA adapters should have gradients
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert param.grad is not None, (
                    f"LoRA param {name} missing gradient"
                )

        # Head should have gradients
        assert head.weight.grad is not None


class TestBetaNonDefault:
    """Verify non-default beta produces different output in combined injection."""

    def test_different_beta_different_output(self) -> None:
        """Combined model with beta=0.5*default should differ from beta=default."""
        torch.manual_seed(42)
        x = torch.randint(0, 1000, (2, 10))

        d_head = 64
        default_beta = 1.0 / (d_head ** 0.5)

        model_default = load_gpt2()
        inject_hopfield(model_default, HopfieldConfig(beta=default_beta))
        inject_lora(model_default, LoRAConfig(r=4, alpha=4.0))
        model_default.eval()

        model_soft = load_gpt2()
        inject_hopfield(model_soft, HopfieldConfig(beta=default_beta * 0.5))
        inject_lora(model_soft, LoRAConfig(r=4, alpha=4.0))
        model_soft.eval()

        with torch.no_grad():
            out_default = model_default(x).last_hidden_state
            out_soft = model_soft(x).last_hidden_state

        assert not torch.allclose(out_default, out_soft, atol=1e-3)
        assert torch.isfinite(out_soft).all()
        assert out_soft.shape == (2, 10, 768)
