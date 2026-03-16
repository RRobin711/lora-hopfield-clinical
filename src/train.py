"""
Shared training loop for GPT-2 classification on DREADDIT.

This single training module serves all four comparison configurations:
frozen baseline, LoRA (various ranks), Hopfield attention replacement, and
full fine-tuning. The loop is model-agnostic — it receives a model with
some parameters frozen and some trainable, and optimizes only the trainable
ones. This is why the optimizer is initialized with
    filter(lambda p: p.requires_grad, model.parameters())
NOT model.parameters() — after LoRA injection, base GPT-2 weights have
requires_grad=False, and including them in AdamW wastes memory on momentum
and variance buffers for parameters that will never update.

Classification head: a single nn.Linear(hidden_size, num_labels) applied to
the hidden state of the last *real* token (the last non-pad position, found
via attention_mask.sum(-1) - 1). GPT-2 is causal — the last real token has
attended to all previous tokens, so its hidden state encodes the full input.
Taking the last sequence position instead would give a pad token hidden state,
which is a common GPT-2 classification bug.

Loss function: nn.CrossEntropyLoss on raw logits — it applies log-softmax
internally for numerical stability. No manual softmax before the loss.

W&B integration: every run logs full config, per-step train loss/lr,
per-epoch val loss/accuracy/f1, and final test metrics. Run names encode
the experiment for the Day 15-16 ablation study (e.g., "lora-r4-s42").
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluate import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyperparameters.

    These are the knobs for the comparison study. The defaults are tuned for
    GPT-2 small on DREADDIT with 8GB VRAM.

    Attributes:
        lr: Peak learning rate after warmup. 2e-5 is standard for fine-tuning
            transformer classifiers (Devlin et al., 2018).
        num_epochs: Maximum training epochs. Early stopping may terminate sooner.
        warmup_fraction: Fraction of total steps for linear LR warmup.
        max_grad_norm: Gradient clipping threshold. 1.0 prevents exploding
            gradients from long Reddit posts with unusual token patterns.
        patience: Early stopping patience — how many epochs of no val loss
            improvement before stopping. 3 is conservative enough to avoid
            stopping on noise, aggressive enough to save compute.
        seed: Random seed for full reproducibility.
        checkpoint_dir: Where to save model checkpoints.
        run_name: W&B run name. Should encode the experiment:
            "lora-r4-s42", "frozen-baseline-s42", "full-finetune-s42".
        wandb_project: W&B project name.
        wandb_enabled: Set False for tests or offline runs.
        num_labels: Number of classification labels. 2 for DREADDIT.
    """

    lr: float = 2e-5
    num_epochs: int = 10
    warmup_fraction: float = 0.1
    max_grad_norm: float = 1.0
    patience: int = 3
    seed: int = 42
    checkpoint_dir: str = "results/checkpoints"
    run_name: str = "default"
    wandb_project: str = "lora-hopfield-clinical"
    wandb_enabled: bool = True
    num_labels: int = 2


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Covers Python stdlib, numpy, PyTorch CPU, and PyTorch CUDA (ROCm presents
    as CUDA). Also sets deterministic cuDNN mode — slightly slower but
    guarantees bitwise reproducibility across runs.

    Args:
        seed: The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_linear_warmup_cosine_scheduler(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Linear warmup followed by cosine decay to zero.

    Standard transformer fine-tuning schedule. The warmup prevents early
    gradient instability (especially important for LoRA where the adapter
    starts at zero and initial gradients can be noisy).

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Steps for linear warmup from 0 to peak lr.
        num_training_steps: Total training steps (warmup + decay).

    Returns:
        LambdaLR scheduler that should be stepped once per training batch.
    """
    import math

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup: 0 -> 1
            return current_step / max(1, num_warmup_steps)
        # Cosine decay: 1 -> 0
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def extract_last_hidden_state(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract the hidden state of the last real (non-pad) token per sequence.

    GPT-2 is causal — the last real token has attended to all prior tokens,
    making its hidden state the natural sequence representation. With right-
    padding (GPT-2 default), pad tokens are at the end, so the last real
    token index is attention_mask.sum(-1) - 1.

    Args:
        hidden_states: Transformer output, shape (B, seq_len, hidden_size).
        attention_mask: Padding mask, shape (B, seq_len). 1 = real, 0 = pad.

    Returns:
        Hidden states at last real positions, shape (B, hidden_size).
    """
    # attention_mask.sum(-1) gives the count of real tokens per row;
    # subtract 1 to get the 0-indexed position of the last real token
    last_real_idx = attention_mask.sum(dim=-1) - 1  # (B,)
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, last_real_idx]  # (B, hidden_size)


@torch.no_grad()
def _evaluate_epoch(
    model: nn.Module,
    head: nn.Linear,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, dict]:
    """Run one evaluation pass over a DataLoader.

    Args:
        model: GPT-2 backbone (in eval mode).
        head: Classification head.
        loader: Validation or test DataLoader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Tuple of (average loss, metrics dict from compute_metrics).
    """
    model.eval()
    head.eval()

    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = extract_last_hidden_state(outputs.last_hidden_state, attention_mask)
        logits = head(pooled)

        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    return avg_loss, metrics


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: TrainConfig,
    hidden_size: int = 768,
) -> dict:
    """Run the full training loop: train, validate, early-stop, test.

    This function is model-agnostic — it receives a model with some parameters
    frozen and some trainable, and optimizes only the trainable ones. The
    classification head is created here (not passed in) because it's always
    freshly initialized per experiment.

    Args:
        model: GPT-2 backbone (with LoRA, Hopfield, or vanilla — doesn't matter).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader (used once after training completes).
        config: TrainConfig with all hyperparameters.
        hidden_size: GPT-2 hidden dimension. 768 for gpt2-small.

    Returns:
        Results dict with training history, final test metrics, and parameter
        counts. JSON-serializable for the aggregation script.
    """
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    model = model.to(device)

    # Classification head: fresh linear layer, always trainable
    head = nn.Linear(hidden_size, config.num_labels).to(device)

    # Optimizer on trainable parameters ONLY — critical for LoRA where base
    # weights are frozen. Including frozen params wastes AdamW state memory.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params += list(head.parameters())
    optimizer = AdamW(trainable_params, lr=config.lr)

    # Scheduler: linear warmup + cosine decay, stepped per batch
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_fraction)
    scheduler = _get_linear_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    criterion = nn.CrossEntropyLoss()

    # Parameter accounting for the results table
    total_params = sum(p.numel() for p in model.parameters()) + sum(
        p.numel() for p in head.parameters()
    )
    trainable_count = sum(p.numel() for p in trainable_params)
    trainable_pct = 100.0 * trainable_count / total_params
    logger.info(
        "Parameters: %d total, %d trainable (%.2f%%)",
        total_params,
        trainable_count,
        trainable_pct,
    )

    # W&B setup
    if config.wandb_enabled:
        import wandb

        wandb.init(
            project=config.wandb_project,
            name=config.run_name,
            config={
                **asdict(config),
                "total_params": total_params,
                "trainable_params": trainable_count,
                "trainable_pct": trainable_pct,
                "device": str(device),
            },
        )

    # Early stopping state
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    global_step = 0

    for epoch in range(config.num_epochs):
        model.train()
        head.train()
        epoch_loss = 0.0
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = extract_last_hidden_state(
                outputs.last_hidden_state, attention_mask
            )
            logits = head(pooled)

            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping on trainable params only
            nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)

            optimizer.step()
            scheduler.step()

            step_loss = loss.item()
            epoch_loss += step_loss
            num_batches += 1
            global_step += 1

            progress.set_postfix(loss=f"{step_loss:.4f}")

            # Per-step W&B logging: loss and learning rate
            if config.wandb_enabled:
                import wandb

                wandb.log(
                    {
                        "train/loss": step_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

        avg_train_loss = epoch_loss / num_batches

        # Validation
        val_loss, val_metrics = _evaluate_epoch(
            model, head, val_loader, criterion, device
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
        }
        history.append(epoch_record)

        logger.info(
            "Epoch %d: train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f",
            epoch + 1,
            avg_train_loss,
            val_loss,
            val_metrics["accuracy"],
            val_metrics["f1_macro"],
        )

        # Per-epoch W&B logging
        if config.wandb_enabled:
            import wandb

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/epoch_loss": avg_train_loss,
                    "val/loss": val_loss,
                    "val/accuracy": val_metrics["accuracy"],
                    "val/f1_macro": val_metrics["f1_macro"],
                },
                step=global_step,
            )

        # Early stopping on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            # Save best checkpoint
            checkpoint_path = checkpoint_dir / f"{config.run_name}_best.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "head_state_dict": head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "config": asdict(config),
                },
                checkpoint_path,
            )
            logger.info("Saved best checkpoint to %s", checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d, best_val_loss=%.4f)",
                    epoch + 1,
                    config.patience,
                    best_val_loss,
                )
                break

    # Load best checkpoint for final test evaluation
    best_checkpoint_path = checkpoint_dir / f"{config.run_name}_best.pt"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        head.load_state_dict(checkpoint["head_state_dict"])
        logger.info("Loaded best checkpoint from epoch %d", checkpoint["epoch"])

    # Final test evaluation
    test_loss, test_metrics = _evaluate_epoch(
        model, head, test_loader, criterion, device
    )
    logger.info(
        "Test results: loss=%.4f acc=%.4f f1=%.4f",
        test_loss,
        test_metrics["accuracy"],
        test_metrics["f1_macro"],
    )

    if config.wandb_enabled:
        import wandb

        wandb.log(
            {
                "test/loss": test_loss,
                "test/accuracy": test_metrics["accuracy"],
                "test/f1_macro": test_metrics["f1_macro"],
            },
            step=global_step,
        )
        wandb.finish()

    results = {
        "run_name": config.run_name,
        "seed": config.seed,
        "total_params": total_params,
        "trainable_params": trainable_count,
        "trainable_pct": trainable_pct,
        "epochs_trained": len(history),
        "best_val_loss": best_val_loss,
        "history": history,
        "test_loss": test_loss,
        "test_accuracy": test_metrics["accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "test_per_class": test_metrics["per_class"],
    }
    return results
