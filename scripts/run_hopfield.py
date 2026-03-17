"""
Single Hopfield attention training run on DREADDIT with GPT-2.

Day 17 deliverable: train GPT-2 with Hopfield attention injection (all 12 blocks,
default beta = 1/sqrt(64), num_iters=1) on DREADDIT binary stress classification.

At default beta, Hopfield retrieval IS standard scaled dot-product attention
(Ramsauer et al., 2020) — so training outcomes should closely match the frozen
baseline (both freeze all base weights, only the classification head trains).
The value is establishing the Hopfield injection infrastructure for Day 18's
comparison table and for future beta exploration.

Results are saved to results/hopfield_s{seed}.json with the same schema as the
ablation JSONs for direct comparison by aggregate_results.py.

Usage:
    python scripts/run_hopfield.py --seed 42
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
from src.hopfield import HopfieldConfig
from src.hopfield_gpt2 import inject_hopfield, verify_hopfield_injection
from src.model import load_gpt2
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
    """Verify GPU can forward+backward a Hopfield-injected model on real data.

    Args:
        train_loader: Real DREADDIT DataLoader (uses first batch).
        device: Must be a CUDA device.

    Raises:
        SystemExit: If any check fails.
    """
    logger.info("=" * 60)
    logger.info("GPU SMOKE TEST (Hopfield)")
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
        inject_hopfield(model)
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
        description="Hopfield attention training on DREADDIT with GPT-2."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Hopfield inverse temperature. None = 1/sqrt(d_head) = standard attention.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output path (default: results/hopfield_s{seed}.json).",
    )
    args = parser.parse_args()

    output_path = Path(
        args.output or f"results/hopfield_s{args.seed}.json"
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

    # Configure and inject Hopfield
    hopfield_config = HopfieldConfig(
        beta=args.beta,  # None → 1/sqrt(d_head) → standard attention
        num_iters=1,  # Non-negotiable for causal models
        dropout=0.0,
    )

    run_name = f"hopfield-s{args.seed}"
    logger.info("Starting %s (beta=%s)", run_name, args.beta or "1/sqrt(d_head)")

    start_time = time.time()

    model = load_gpt2()
    inject_hopfield(model, hopfield_config)
    summary = verify_hopfield_injection(model)
    if not summary["all_checks_passed"]:
        print("ERROR: Hopfield injection verification failed.", file=sys.stderr)
        sys.exit(1)

    train_config = TrainConfig(
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
        "rank": "hopfield",
        "seed": args.seed,
        "beta": hopfield_config.beta
        if hopfield_config.beta is not None
        else round(1.0 / (64**0.5), 6),
        "num_iters": hopfield_config.num_iters,
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
    logger.info("HOPFIELD TRAINING COMPLETE")
    logger.info("  Test accuracy: %.4f", result_json["test_accuracy"])
    logger.info("  Test F1 (macro): %.4f", result_json["test_f1_macro"])
    logger.info("  Best epoch: %d", result_json["best_epoch"])
    logger.info("  Training time: %.1f seconds", training_time)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
