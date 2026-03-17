"""
Full fine-tune training run on DREADDIT with GPT-2.

Day 18 deliverable: train GPT-2 with ALL parameters unfrozen on DREADDIT binary
stress classification. This establishes the upper bound for the 4-way comparison
table — the best accuracy achievable when the entire 124M-parameter model adapts
to the task. LoRA's value proposition is recovering most of this performance with
<1% of the trainable parameters.

Results are saved to results/full_finetune_s{seed}.json with the same schema as the
ablation JSONs for direct comparison by aggregate_results.py.

Usage:
    python scripts/run_full_finetune.py --seed 42
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
from src.model import load_gpt2, unfreeze_all_parameters
from src.train import TrainConfig, extract_last_hidden_state, seed_everything, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _write_result_atomic(path: Path, result_dict: dict) -> None:
    """Write JSON atomically: temp file then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".json.tmp", prefix=path.stem
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(result_dict, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def gpu_smoke_test(
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    """Verify GPU can forward+backward a fully unfrozen model on real data.

    Args:
        train_loader: Real DREADDIT DataLoader (uses first batch).
        device: Must be a CUDA device.

    Raises:
        SystemExit: If any check fails.
    """
    logger.info("=" * 60)
    logger.info("GPU SMOKE TEST (Full Fine-Tune)")
    logger.info("=" * 60)

    hsa_version = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if hsa_version != "10.3.0":
        print(
            f"SMOKE TEST FAILED: HSA_OVERRIDE_GFX_VERSION={hsa_version!r}, "
            "expected '10.3.0'.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not torch.cuda.is_available():
        print(
            "SMOKE TEST FAILED: torch.cuda.is_available() returned False.",
            file=sys.stderr,
        )
        sys.exit(1)
    logger.info("GPU: %s", torch.cuda.get_device_name(0))

    try:
        model = load_gpt2()
        unfreeze_all_parameters(model)
        model = model.to(device)
        head = nn.Linear(model.config.hidden_size, 2).to(device)
    except Exception as exc:
        print(f"SMOKE TEST FAILED: model setup error: {exc}", file=sys.stderr)
        sys.exit(1)

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
            print(f"SMOKE TEST FAILED: loss={loss.item()}", file=sys.stderr)
            sys.exit(1)

        loss.backward()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        trainable_params += list(head.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=2e-5)
        optimizer.step()

    except Exception as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    logger.info("Loss: %.4f (finite: True)", loss.item())
    logger.info("GPU SMOKE TEST PASSED")
    logger.info("=" * 60)

    del model, head, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full fine-tune training on DREADDIT with GPT-2."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help=(
            "Peak learning rate. Default 2e-5 matches TrainConfig for backward "
            "compatibility. For full fine-tune on small datasets, 5e-6 is "
            "recommended (Mosbach et al., 2021) — 2e-5 causes memorization "
            "in <3 epochs with 124M unfrozen params on ~2800 samples."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output path (default: results/full_finetune_s{seed}_lr{lr}.json).",
    )
    args = parser.parse_args()

    # Format lr for filenames: 5e-06 -> "5e-06", 2e-05 -> "2e-05"
    lr_tag = f"{args.lr:.0e}"
    output_path = Path(
        args.output or f"results/full_finetune_s{args.seed}_lr{lr_tag}.json"
    )

    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/checkpoints").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data once
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

    # GPU smoke test
    if device.type == "cuda":
        gpu_smoke_test(train_loader, device)
    else:
        print(
            "ERROR: No CUDA device available. This script requires GPU.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_name = f"full-finetune-s{args.seed}-lr{lr_tag}"
    logger.info("Starting %s (all parameters unfrozen, lr=%s)", run_name, args.lr)

    start_time = time.time()

    model = load_gpt2()
    unfreeze_all_parameters(model)

    train_config = TrainConfig(
        lr=args.lr,
        seed=args.seed,
        run_name=run_name,
        wandb_project="lora-hopfield-clinical",
        wandb_enabled=True,
    )

    raw_results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=train_config,
    )

    training_time = time.time() - start_time

    result_json = {
        "rank": "full",
        "seed": args.seed,
        "lr": args.lr,
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

    # Free VRAM
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info("FULL FINE-TUNE TRAINING COMPLETE")
    logger.info("  Test accuracy: %.4f", result_json["test_accuracy"])
    logger.info("  Test F1 (macro): %.4f", result_json["test_f1_macro"])
    logger.info("  Best epoch: %d", result_json["best_epoch"])
    logger.info("  Training time: %.1f seconds", training_time)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
