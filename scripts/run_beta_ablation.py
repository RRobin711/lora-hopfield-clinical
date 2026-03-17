"""
Hopfield beta ablation study on DREADDIT with GPT-2.

Ablates the inverse temperature beta across multipliers of the default value
(1/sqrt(d_head) = 1/sqrt(64) = 0.125 for GPT-2 small). Beta controls the
sharpness of the Hopfield energy landscape:
    - Low beta (0.5x) = softer, more distributed attention across stored patterns
    - Default beta (1.0x) = standard scaled dot-product attention (Ramsauer et al.)
    - High beta (4.0x) = sharp, winner-take-all retrieval

Phase 1 connection: Phase 1 found that HyDE helped on informal layperson queries
by bridging vocabulary gaps — softer matching improved retrieval when query and
corpus had a terminology mismatch. DREADDIT is informal Reddit text hitting a
classification boundary. Lower beta = softer, more distributed attention =
potentially better at capturing distributed stress signals in informal language
where no single token is a strong indicator. This ablation tests that hypothesis
directly: does relaxing the energy landscape improve stress detection on informal
text, or does sharper retrieval help by suppressing noise?

Resumability: if a result file already exists with valid JSON, that multiplier is
skipped. The 1.0x run reuses results/hopfield_s{seed}.json if available.

Usage:
    # Run full beta ablation:
    python scripts/run_beta_ablation.py --seed 42

    # Run specific multipliers:
    python scripts/run_beta_ablation.py --multipliers 0.5 2.0 --seed 42

Reference: Ramsauer et al. (2020), https://arxiv.org/abs/2008.02217
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
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

# GPT-2 small: d_head = hidden_size / num_heads = 768 / 12 = 64
D_HEAD = 64
DEFAULT_BETA = 1.0 / (D_HEAD ** 0.5)  # 0.125


def _output_path_for_multiplier(
    multiplier: float, seed: int, output_dir: str,
) -> Path:
    """Determine the output JSON path for a given beta multiplier."""
    return Path(output_dir) / f"hopfield_beta{multiplier}x_s{seed}.json"


def _is_already_complete(path: Path) -> bool:
    """Check if a result file exists and is valid JSON with required keys."""
    if not path.exists():
        return False
    try:
        with open(path) as f:
            content = json.load(f)
        required = {"rank", "seed", "test_accuracy", "test_f1_macro", "beta"}
        return required.issubset(content.keys())
    except (json.JSONDecodeError, KeyError):
        return False


def _write_result_atomic(path: Path, result_dict: dict) -> None:
    """Write JSON atomically: temp file in same directory, then rename."""
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


def _try_copy_existing_default(
    seed: int, output_dir: str,
) -> Path | None:
    """If results/hopfield_s{seed}.json exists, return its path for copying.

    The 1.0x multiplier is identical to the default Hopfield run. Reusing it
    saves ~30 min of GPU time.
    """
    existing = Path(f"results/hopfield_s{seed}.json")
    if not existing.exists():
        return None
    try:
        with open(existing) as f:
            content = json.load(f)
        required = {"rank", "seed", "test_accuracy", "test_f1_macro"}
        if required.issubset(content.keys()):
            return existing
    except (json.JSONDecodeError, KeyError):
        pass
    return None


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
    logger.info("GPU SMOKE TEST (Hopfield Beta Ablation)")
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

    # Test with non-default beta to exercise the actual ablation code path
    try:
        model = load_gpt2()
        hopfield_config = HopfieldConfig(beta=DEFAULT_BETA * 2.0, num_iters=1)
        inject_hopfield(model, hopfield_config)
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


def run_single_beta(
    multiplier: float,
    seed: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    output_path: Path,
) -> None:
    """Train one Hopfield configuration at a given beta multiplier.

    Loads a fresh GPT-2, injects Hopfield with beta = multiplier * default_beta,
    trains, writes results JSON, and frees VRAM.

    Args:
        multiplier: Beta multiplier relative to 1/sqrt(d_head).
        seed: Random seed for reproducibility.
        train_loader: Training DataLoader (pre-loaded, shared across runs).
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader.
        output_path: Where to write the results JSON.
    """
    actual_beta = DEFAULT_BETA * multiplier
    run_name = f"hopfield-beta{multiplier}x-s{seed}"
    logger.info(
        "Starting %s (beta=%.6f = %.1fx * %.6f)",
        run_name, actual_beta, multiplier, DEFAULT_BETA,
    )

    start_time = time.time()

    model = load_gpt2()
    hopfield_config = HopfieldConfig(
        beta=actual_beta,
        num_iters=1,
        dropout=0.0,
    )
    inject_hopfield(model, hopfield_config)
    summary = verify_hopfield_injection(model)
    if not summary["all_checks_passed"]:
        raise RuntimeError(
            f"Hopfield injection verification failed for beta={actual_beta}"
        )

    train_config = TrainConfig(
        seed=seed,
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
        "config": "hopfield",
        "seed": seed,
        "beta": actual_beta,
        "beta_multiplier": multiplier,
        "d_head": D_HEAD,
        "num_iters": 1,
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

    # Free VRAM before next run
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hopfield beta ablation on DREADDIT with GPT-2."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory for result JSON files (default: results).",
    )
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 4.0],
        help="Beta multipliers relative to 1/sqrt(d_head) (default: 0.5 1.0 2.0 4.0).",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path("results/checkpoints").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data ONCE — tokenization on HDD is slow, reuse across all betas
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

    logger.info("Beta ablation: multipliers=%s, default_beta=%.6f", args.multipliers, DEFAULT_BETA)
    logger.info("Actual beta values: %s", [round(DEFAULT_BETA * m, 6) for m in args.multipliers])

    completed = 0
    skipped = 0
    failed = 0

    for multiplier in args.multipliers:
        output_path = _output_path_for_multiplier(
            multiplier, args.seed, args.output_dir,
        )

        # Resumability: skip if already complete
        if _is_already_complete(output_path):
            logger.info(
                "Beta %.1fx already complete (%s exists), skipping",
                multiplier, output_path,
            )
            skipped += 1
            continue

        # 1.0x optimization: copy existing default Hopfield run if available
        if multiplier == 1.0:
            existing_path = _try_copy_existing_default(args.seed, args.output_dir)
            if existing_path is not None:
                logger.info(
                    "Beta 1.0x: copying existing %s to %s (saves ~30 min)",
                    existing_path, output_path,
                )
                # Read, augment with ablation-specific fields, write
                with open(existing_path) as f:
                    existing_result = json.load(f)
                existing_result["beta_multiplier"] = 1.0
                existing_result["d_head"] = D_HEAD
                existing_result["config"] = "hopfield"
                if "beta" not in existing_result:
                    existing_result["beta"] = DEFAULT_BETA
                if "num_iters" not in existing_result:
                    existing_result["num_iters"] = 1
                _write_result_atomic(output_path, existing_result)
                completed += 1
                continue

        try:
            run_single_beta(
                multiplier=multiplier,
                seed=args.seed,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                output_path=output_path,
            )
            completed += 1
        except Exception as exc:
            logger.error(
                "Beta %.1fx FAILED: %s", multiplier, exc, exc_info=True,
            )
            error_result = {
                "rank": "hopfield",
                "config": "hopfield",
                "seed": args.seed,
                "beta_multiplier": multiplier,
                "beta": DEFAULT_BETA * multiplier,
                "error": str(exc),
            }
            _write_result_atomic(output_path, error_result)
            failed += 1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    logger.info("=" * 60)
    logger.info("BETA ABLATION COMPLETE")
    logger.info("  Completed: %d", completed)
    logger.info("  Skipped (already done): %d", skipped)
    logger.info("  Failed: %d", failed)
    logger.info("=" * 60)

    if failed > 0:
        logger.warning(
            "Some multipliers failed. Re-run to retry (completed runs "
            "will be skipped)."
        )


if __name__ == "__main__":
    main()
