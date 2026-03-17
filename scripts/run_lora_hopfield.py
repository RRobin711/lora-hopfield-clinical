"""
Combined LoRA + Hopfield attention training on DREADDIT with GPT-2.

This script injects BOTH LoRA adapters (learnable low-rank projections) and
Hopfield attention (energy-minimizing retrieval) into the same GPT-2 model.
LoRA modifies *what* gets projected (adapts the Q/K/V weight matrices);
Hopfield modifies *how* the projected Q/K/V interact (replaces softmax attention
with Hopfield retrieval at a configurable inverse temperature beta).

Injection order is critical — Hopfield FIRST, then LoRA:

    1. inject_hopfield() freezes all params, replaces attn.forward with a custom
       method that reads self.c_attn dynamically at call time.
    2. inject_lora() freezes all params (already frozen — no-op), then replaces
       attn.c_attn Conv1D with LoRALinear via setattr. The LoRA adapter params
       (lora_A, lora_B) are created with requires_grad=True.
    3. At forward time, the Hopfield forward reads self.c_attn → finds the
       LoRALinear → gets adapted QKV projections → routes through Hopfield retrieval.

The reverse order (LoRA first, Hopfield second) would fail because
inject_hopfield() freezes ALL parameters, including the LoRA adapters — making
them non-trainable and defeating the purpose.

Why this combination matters: LoRA alone adapts the projection weights to the
task but uses standard attention. Hopfield alone changes the attention mechanism
but with frozen projections. The combination tests whether adapting both the
weight space and the attention mechanism produces better stress detection than
either alone — particularly on informal Reddit text where distributed attention
patterns (Hopfield) may complement learned task-specific projections (LoRA).

Usage:
    python scripts/run_lora_hopfield.py --seed 42 --rank 8 --beta_multiplier 1.0
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
from src.lora import LoRAConfig
from src.model import inject_lora, load_gpt2, verify_lora_injection
from src.train import TrainConfig, extract_last_hidden_state, seed_everything, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# GPT-2 small: d_head = hidden_size / num_heads = 768 / 12 = 64
D_HEAD = 64
DEFAULT_BETA = 1.0 / (D_HEAD ** 0.5)  # 0.125


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


def gpu_smoke_test(
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    """Verify GPU can forward+backward a combined LoRA+Hopfield model.

    Exercises the exact injection order used in training: Hopfield first,
    then LoRA. Validates that LoRA adapters are trainable after both injections.

    Args:
        train_loader: Real DREADDIT DataLoader (uses first batch).
        device: Must be a CUDA device.

    Raises:
        SystemExit: If any check fails.
    """
    logger.info("=" * 60)
    logger.info("GPU SMOKE TEST (LoRA + Hopfield)")
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

        # Injection order: Hopfield FIRST, then LoRA
        hopfield_config = HopfieldConfig(beta=DEFAULT_BETA, num_iters=1)
        inject_hopfield(model, hopfield_config)

        lora_config = LoRAConfig(r=4, alpha=4.0)
        inject_lora(model, lora_config)

        model = model.to(device)
        head = nn.Linear(model.config.hidden_size, 2).to(device)
    except Exception as exc:
        print(f"SMOKE TEST FAILED: model setup error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Verify LoRA adapters are trainable after combined injection
    lora_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    if lora_trainable == 0:
        print(
            "SMOKE TEST FAILED: no trainable model params after combined injection. "
            "LoRA adapters were likely frozen by inject_hopfield().",
            file=sys.stderr,
        )
        sys.exit(1)
    logger.info("LoRA trainable params: %d (expected >0)", lora_trainable)

    # Verify Hopfield beta buffers exist
    for block_idx, block in enumerate(model.h):
        if not hasattr(block.attn, "hopfield_attn"):
            print(
                f"SMOKE TEST FAILED: block {block_idx} missing hopfield_attn",
                file=sys.stderr,
            )
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
        description="Combined LoRA + Hopfield training on DREADDIT with GPT-2."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--rank", type=int, default=8,
        help="LoRA rank (default: 8).",
    )
    parser.add_argument(
        "--beta_multiplier", type=float, default=1.0,
        help="Beta multiplier relative to 1/sqrt(d_head) (default: 1.0).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory for result JSON files (default: results).",
    )
    args = parser.parse_args()

    actual_beta = DEFAULT_BETA * args.beta_multiplier
    run_name = f"lora-r{args.rank}-hopfield-beta{args.beta_multiplier}x-s{args.seed}"
    output_path = (
        Path(args.output_dir)
        / f"lora_r{args.rank}_hopfield_beta{args.beta_multiplier}x_s{args.seed}.json"
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
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

    logger.info(
        "Starting %s (rank=%d, beta=%.6f = %.1fx * %.6f)",
        run_name, args.rank, actual_beta, args.beta_multiplier, DEFAULT_BETA,
    )

    start_time = time.time()

    # Fresh model — injection order: Hopfield FIRST, then LoRA
    model = load_gpt2()

    # Step 1: Hopfield injection — replaces attn.forward, freezes all params
    hopfield_config = HopfieldConfig(
        beta=actual_beta,
        num_iters=1,
        dropout=0.0,
    )
    inject_hopfield(model, hopfield_config)
    hop_summary = verify_hopfield_injection(model)
    if not hop_summary["all_checks_passed"]:
        print("ERROR: Hopfield injection verification failed.", file=sys.stderr)
        sys.exit(1)
    logger.info("Hopfield injection verified: %d blocks", hop_summary["blocks_with_hopfield"])

    # Step 2: LoRA injection — replaces attn.c_attn Conv1D with LoRALinear,
    # makes adapter params trainable. The Hopfield forward reads self.c_attn
    # dynamically, so it will pick up the LoRALinear at call time.
    # alpha=rank keeps effective adapter scale constant across rank choices.
    lora_config = LoRAConfig(r=args.rank, alpha=float(args.rank))
    inject_lora(model, lora_config)
    lora_summary = verify_lora_injection(model, lora_config)
    if not lora_summary["all_checks_passed"]:
        print("ERROR: LoRA injection verification failed.", file=sys.stderr)
        sys.exit(1)
    logger.info(
        "LoRA injection verified: %d modules, %d trainable params (%.4f%%)",
        len(lora_summary["lora_modules"]),
        lora_summary["trainable_params"],
        lora_summary["trainable_pct"],
    )

    # Confirm Hopfield is still active after LoRA injection
    for block_idx, block in enumerate(model.h):
        if not hasattr(block.attn, "hopfield_attn"):
            print(
                f"ERROR: Hopfield lost on block {block_idx} after LoRA injection.",
                file=sys.stderr,
            )
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
        "config": "lora+hopfield",
        "rank": args.rank,
        "seed": args.seed,
        "beta": actual_beta,
        "beta_multiplier": args.beta_multiplier,
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

    # Free VRAM
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("=" * 60)
    logger.info("LORA + HOPFIELD TRAINING COMPLETE")
    logger.info("  Test accuracy: %.4f", result_json["test_accuracy"])
    logger.info("  Test F1 (macro): %.4f", result_json["test_f1_macro"])
    logger.info("  Best epoch: %d", result_json["best_epoch"])
    logger.info("  Training time: %.1f seconds", training_time)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
