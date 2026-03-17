"""
LoRA rank ablation study on DREADDIT with GPT-2.

Runs ranks r in {1, 4, 8, 16, 32} plus a frozen baseline (rank=0) sequentially,
saving per-rank results to results/ablation_r{r}.json (or ablation_frozen.json).
Designed for the RX 6700S (8GB VRAM): each rank loads a fresh model, trains,
saves results, then explicitly frees VRAM before the next rank.

GPU smoke test runs first — if the GPU cannot complete a forward+backward pass
with a real DREADDIT batch, the script exits immediately. No silent CPU fallback.

Resumability: if a result file already exists and contains valid JSON, that rank
is skipped. This lets the script recover from mid-ablation crashes without
re-running completed ranks.

Usage:
    # Run full ablation (all ranks + frozen baseline):
    python scripts/run_lora_ablation.py --seed 42

    # Run single rank (for claude-flow sub-agents):
    python scripts/run_lora_ablation.py --rank 4 --seed 42

    # Override output path (single-rank mode only):
    python scripts/run_lora_ablation.py --rank 4 --seed 42 --output results/ablation_r4.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

from src.data import DataConfig, load_dreaddit
from src.lora import LoRAConfig
from src.model import (
    freeze_all_parameters,
    inject_lora,
    load_gpt2,
    verify_lora_injection,
)
from src.train import TrainConfig, extract_last_hidden_state, seed_everything, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Frozen baseline (rank=0) runs first so interrupted ablations still have a
# lower bound reference in the W&B comparison chart
ALL_RANKS = (0, 1, 4, 8, 16, 32, 64, 128)


def _result_path_for_rank(rank: int, output_override: str | None = None) -> Path:
    """Determine the output JSON path for a given rank."""
    if output_override is not None:
        return Path(output_override)
    if rank == 0:
        return Path("results/ablation_frozen.json")
    return Path(f"results/ablation_r{rank}.json")


def _is_already_complete(path: Path) -> bool:
    """Check if a result file exists and is valid JSON with required keys."""
    if not path.exists():
        return False
    try:
        with open(path) as f:
            content = json.load(f)
        required = {"rank", "seed", "test_accuracy", "test_f1_macro"}
        return required.issubset(content.keys())
    except (json.JSONDecodeError, KeyError):
        return False


def _write_result_atomic(path: Path, result_dict: dict) -> None:
    """Write JSON atomically: temp file in same directory, then rename.

    Prevents partial files if the process is killed mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".json.tmp", prefix=path.stem
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(result_dict, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up temp file if rename failed
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def gpu_smoke_test(
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    """Verify GPU can run a full forward+backward+step on real data.

    Loads GPT-2, injects LoRA r=4, runs one batch through the exact same
    code path used in training. If anything fails — model loading, data
    transfer, forward, backward, optimizer step, loss finiteness — the
    script dies with a clear error. No silent CPU fallback.

    Args:
        train_loader: Real DREADDIT DataLoader (uses first batch).
        device: Must be a CUDA device.

    Raises:
        SystemExit: If any check fails.
    """
    logger.info("=" * 60)
    logger.info("GPU SMOKE TEST")
    logger.info("=" * 60)

    # Check 1: HSA_OVERRIDE_GFX_VERSION
    hsa_version = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if hsa_version != "10.3.0":
        print(
            f"SMOKE TEST FAILED: HSA_OVERRIDE_GFX_VERSION={hsa_version!r}, "
            "expected '10.3.0'. ROCm will not detect the RX 6700S.",
            file=sys.stderr,
        )
        sys.exit(1)
    logger.info("HSA_OVERRIDE_GFX_VERSION=%s", hsa_version)

    # Check 2: CUDA available
    if not torch.cuda.is_available():
        print(
            "SMOKE TEST FAILED: torch.cuda.is_available() returned False. "
            "Check ROCm installation and HSA_OVERRIDE_GFX_VERSION.",
            file=sys.stderr,
        )
        sys.exit(1)
    logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Check 3: load model, inject LoRA, move to GPU
    try:
        model = load_gpt2()
        lora_config = LoRAConfig(r=4, alpha=4.0)
        inject_lora(model, lora_config)
        model = model.to(device)
        head = nn.Linear(model.config.hidden_size, 2).to(device)
    except Exception as exc:
        print(
            f"SMOKE TEST FAILED: could not load/inject/move model to GPU: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check 4: forward + backward + step on real batch
    try:
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = extract_last_hidden_state(outputs.last_hidden_state, attention_mask)
        logits = head(pooled)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            print(
                f"SMOKE TEST FAILED: loss is not finite (loss={loss.item()}).",
                file=sys.stderr,
            )
            sys.exit(1)

        loss.backward()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        trainable_params += list(head.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=2e-5)
        optimizer.step()

    except Exception as exc:
        print(
            f"SMOKE TEST FAILED: forward/backward/step error: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    logger.info("Loss: %.4f (finite: True)", loss.item())
    logger.info("GPU SMOKE TEST PASSED")
    logger.info("=" * 60)

    # Free VRAM — caller's references must be deleted explicitly so
    # refcount drops to zero before gc.collect() runs
    del model
    del head
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()


def run_single_rank(
    rank: int,
    seed: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    output_path: Path,
) -> None:
    """Train one configuration (LoRA at given rank, or frozen baseline if rank=0).

    Loads a fresh GPT-2, configures it, calls train(), writes results JSON,
    and frees VRAM.

    Args:
        rank: LoRA rank (1, 4, 8, 16, 32) or 0 for frozen baseline.
        seed: Random seed for reproducibility.
        train_loader: Training DataLoader (pre-loaded, shared across ranks).
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader.
        output_path: Where to write the results JSON.
    """
    if rank == 0:
        run_name = f"frozen-baseline-s{seed}"
        logger.info("Starting frozen baseline (rank=0)")
    else:
        run_name = f"lora-r{rank}-s{seed}"
        logger.info("Starting LoRA rank=%d", rank)

    start_time = time.time()

    # Fresh model for each rank — no state leakage
    model = load_gpt2()

    if rank == 0:
        # Frozen baseline: all weights frozen, only classification head trains
        freeze_all_parameters(model)
    else:
        # alpha=r so that alpha/r = 1.0 for every rank — this keeps the
        # effective adapter update scale constant across the ablation, isolating
        # rank as the sole independent variable. Without this, higher ranks
        # would also get larger effective learning rates, confounding results.
        lora_config = LoRAConfig(r=rank, alpha=float(rank))
        inject_lora(model, lora_config)
        verify_lora_injection(model, lora_config)

    train_config = TrainConfig(
        seed=seed,
        run_name=run_name,
        wandb_project="lora-hopfield-clinical",
        wandb_enabled=True,
    )

    # train() handles: device placement, head creation, optimizer, scheduling,
    # early stopping, checkpointing, test evaluation, and W&B logging
    raw_results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=train_config,
    )

    training_time = time.time() - start_time

    # Build the results schema required by the architecture plan.
    # wandb_run_id and best_epoch come from train()'s results dict — they
    # are captured inside train() before wandb.finish() clears wandb.run.
    result_json = {
        "rank": rank,
        "seed": seed,
        "trainable_params": raw_results["trainable_params"],
        "total_params": raw_results["total_params"],
        "trainable_pct": raw_results["trainable_pct"],
        "test_accuracy": raw_results["test_accuracy"],
        "test_f1_macro": raw_results["test_f1_macro"],
        "best_val_loss": raw_results["best_val_loss"],
        "best_epoch": raw_results["best_epoch"],
        "total_training_time_s": round(training_time, 1),
        "wandb_run_id": raw_results["wandb_run_id"],
    }

    _write_result_atomic(output_path, result_json)
    logger.info("Results written to %s", output_path)

    # Free VRAM before loading the next rank — del the caller's reference
    # so refcount drops to zero before gc.collect()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA rank ablation on DREADDIT with GPT-2."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        choices=[0, 1, 4, 8, 16, 32, 64, 128],
        help="Run a single rank only. 0 = frozen baseline. Omit for full ablation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output path (single-rank mode only).",
    )
    args = parser.parse_args()

    if args.output is not None and args.rank is None:
        parser.error("--output can only be used with --rank")

    # Ensure results directory exists
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/checkpoints").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data ONCE — tokenization on HDD is slow, reuse across all ranks
    logger.info("Loading DREADDIT dataset and tokenizing...")
    seed_everything(args.seed)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    data_config = DataConfig(seed=args.seed)
    train_loader, val_loader, test_loader = load_dreaddit(data_config, tokenizer)
    logger.info(
        "Data loaded: %d train, %d val, %d test batches",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )

    # GPU smoke test — must pass before any training
    if device.type == "cuda":
        gpu_smoke_test(train_loader, device)
    else:
        print(
            "ERROR: No CUDA device available. This script requires GPU. "
            "Check HSA_OVERRIDE_GFX_VERSION and ROCm setup.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine which ranks to run
    ranks_to_run = (args.rank,) if args.rank is not None else ALL_RANKS

    completed = 0
    skipped = 0
    failed = 0

    for rank in ranks_to_run:
        output_path = _result_path_for_rank(rank, args.output)

        # Resumability: skip if already complete
        if _is_already_complete(output_path):
            logger.info(
                "Rank %d already complete (%s exists), skipping", rank, output_path
            )
            skipped += 1
            continue

        try:
            run_single_rank(
                rank=rank,
                seed=args.seed,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                output_path=output_path,
            )
            completed += 1
        except Exception as exc:
            logger.error("Rank %d FAILED: %s", rank, exc, exc_info=True)
            # Write error state so aggregate_results.py can report it
            error_result = {
                "rank": rank,
                "seed": args.seed,
                "error": str(exc),
            }
            _write_result_atomic(output_path, error_result)
            failed += 1
            # Free any leaked VRAM before continuing
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    logger.info("=" * 60)
    logger.info("ABLATION COMPLETE")
    logger.info("  Completed: %d", completed)
    logger.info("  Skipped (already done): %d", skipped)
    logger.info("  Failed: %d", failed)
    logger.info("=" * 60)

    if failed > 0:
        logger.warning(
            "Some ranks failed. Re-run the script to retry (completed ranks "
            "will be skipped). Or run a single rank: --rank <r>"
        )

    # Print aggregation hint
    logger.info(
        "To aggregate results: python scripts/aggregate_results.py"
    )


if __name__ == "__main__":
    main()
