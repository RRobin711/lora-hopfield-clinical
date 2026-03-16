"""Tests for training infrastructure.

Tests the pure-logic components of the training loop without requiring GPU,
HuggingFace model downloads, or W&B connections. Focuses on:
- seed_everything reproducibility
- extract_last_hidden_state correctness with padding
- LR scheduler shape (warmup + cosine decay)
- Optimizer parameter filtering
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.train import (
    TrainConfig,
    _get_linear_warmup_cosine_scheduler,
    extract_last_hidden_state,
    seed_everything,
)


class TestSeedEverything:
    """Verify reproducibility across seed_everything calls."""

    def test_torch_reproducible(self) -> None:
        seed_everything(42)
        a = torch.randn(10)
        seed_everything(42)
        b = torch.randn(10)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self) -> None:
        seed_everything(42)
        a = torch.randn(10)
        seed_everything(99)
        b = torch.randn(10)
        assert not torch.equal(a, b)


class TestExtractLastHiddenState:
    """Verify correct extraction of last real token hidden state."""

    def test_no_padding(self) -> None:
        """All tokens real — last position is the last real token."""
        hidden = torch.randn(2, 5, 8)
        mask = torch.ones(2, 5, dtype=torch.long)
        extracted = extract_last_hidden_state(hidden, mask)
        assert extracted.shape == (2, 8)
        # Should be the last position (index 4)
        torch.testing.assert_close(extracted[0], hidden[0, 4])
        torch.testing.assert_close(extracted[1], hidden[1, 4])

    def test_with_padding(self) -> None:
        """Right-padded sequences — must pick last real, not last position."""
        hidden = torch.randn(2, 5, 8)
        # First sequence: 3 real tokens, 2 padding
        # Second sequence: 5 real tokens, 0 padding
        mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ])
        extracted = extract_last_hidden_state(hidden, mask)
        # First: last real is index 2
        torch.testing.assert_close(extracted[0], hidden[0, 2])
        # Second: last real is index 4
        torch.testing.assert_close(extracted[1], hidden[1, 4])

    def test_single_token(self) -> None:
        """Edge case: sequence of length 1 (one real token)."""
        hidden = torch.randn(1, 3, 8)
        mask = torch.tensor([[1, 0, 0]])
        extracted = extract_last_hidden_state(hidden, mask)
        torch.testing.assert_close(extracted[0], hidden[0, 0])

    def test_variable_lengths(self) -> None:
        """Batch with varying real sequence lengths."""
        hidden = torch.randn(3, 6, 4)
        mask = torch.tensor([
            [1, 0, 0, 0, 0, 0],  # 1 real token
            [1, 1, 1, 0, 0, 0],  # 3 real tokens
            [1, 1, 1, 1, 1, 1],  # 6 real tokens
        ])
        extracted = extract_last_hidden_state(hidden, mask)
        torch.testing.assert_close(extracted[0], hidden[0, 0])
        torch.testing.assert_close(extracted[1], hidden[1, 2])
        torch.testing.assert_close(extracted[2], hidden[2, 5])


class TestLRScheduler:
    """Verify linear warmup + cosine decay shape."""

    def test_warmup_starts_at_zero(self) -> None:
        model = nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = _get_linear_warmup_cosine_scheduler(
            optimizer, num_warmup_steps=10, num_training_steps=100
        )
        # At step 0, LR multiplier should be 0
        assert scheduler.get_last_lr()[0] == 0.0

    def test_warmup_reaches_peak(self) -> None:
        model = nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = _get_linear_warmup_cosine_scheduler(
            optimizer, num_warmup_steps=10, num_training_steps=100
        )
        # Step through warmup
        for _ in range(10):
            optimizer.step()
            scheduler.step()
        # At end of warmup, LR should be at peak (multiplier = 1.0)
        assert scheduler.get_last_lr()[0] == 1.0

    def test_cosine_decays_to_zero(self) -> None:
        model = nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = _get_linear_warmup_cosine_scheduler(
            optimizer, num_warmup_steps=10, num_training_steps=100
        )
        # Step through all training steps
        for _ in range(100):
            optimizer.step()
            scheduler.step()
        # At end, LR should be ~0
        assert scheduler.get_last_lr()[0] < 0.01

    def test_warmup_is_monotonically_increasing(self) -> None:
        model = nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = _get_linear_warmup_cosine_scheduler(
            optimizer, num_warmup_steps=20, num_training_steps=100
        )
        prev_lr = -1.0
        for _ in range(20):
            optimizer.step()
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            assert current_lr >= prev_lr, (
                f"LR decreased during warmup: {prev_lr} -> {current_lr}"
            )
            prev_lr = current_lr


class TestOptimizerFiltering:
    """Verify that only trainable parameters are collected for the optimizer.

    This is the critical pattern for LoRA: frozen base weights must NOT be
    in the optimizer, or AdamW wastes memory on their momentum/variance.
    """

    def test_filter_excludes_frozen_params(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        # Freeze first layer
        for p in model[0].parameters():
            p.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
        # Should have 2 param groups entries (weight + bias of second layer)
        total_optimizer_params = sum(
            p.numel() for group in optimizer.param_groups for p in group["params"]
        )
        expected = sum(p.numel() for p in model[1].parameters())
        assert total_optimizer_params == expected

    def test_frozen_params_unchanged_after_step(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        for p in model[0].parameters():
            p.requires_grad = False
        frozen_weight_before = model[0].weight.data.clone()

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)

        # Forward + backward + step
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Frozen weight should be untouched
        assert torch.equal(model[0].weight.data, frozen_weight_before)


class TestTrainConfig:
    """Verify config defaults match CLAUDE.md specification."""

    def test_defaults(self) -> None:
        cfg = TrainConfig()
        assert cfg.lr == 2e-5
        assert cfg.num_epochs == 10
        assert cfg.warmup_fraction == 0.1
        assert cfg.max_grad_norm == 1.0
        assert cfg.patience == 3
        assert cfg.seed == 42
        assert cfg.wandb_project == "lora-hopfield-clinical"
        assert cfg.num_labels == 2
