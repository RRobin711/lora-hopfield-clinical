"""
Aggregate LoRA ablation results into a summary table and CSV.

Reads per-rank JSON files from results/ (written by run_lora_ablation.py),
extracts key metrics, and produces:
    1. A formatted table printed to stdout
    2. results/ablation_summary.csv for downstream analysis

This script has zero model dependencies — no torch, no transformers, no W&B.
It is pure data aggregation over JSON files, safe to run at any time.

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --pattern "results/ablation_*.json"
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

# Expected ranks from the ablation study (0 = frozen baseline)
EXPECTED_RANKS = {0, 1, 4, 8, 16, 32}

# Column spec: (json_key, csv_header, format_string for stdout)
COLUMNS = [
    ("rank", "rank", "{:>4}"),
    ("trainable_params", "trainable_params", "{:>16,}"),
    ("trainable_pct", "trainable_pct", "{:>14.4f}"),
    ("test_accuracy", "test_accuracy", "{:>13.4f}"),
    ("test_f1_macro", "test_f1_macro", "{:>13.4f}"),
    ("best_val_loss", "best_val_loss", "{:>13.4f}"),
]


def _find_result_files(pattern: str) -> list[Path]:
    """Glob for ablation result files, including the frozen baseline."""
    paths = set()
    for p in glob.glob(pattern):
        paths.add(Path(p))
    # Always check for frozen baseline explicitly
    frozen_path = Path("results/ablation_frozen.json")
    if frozen_path.exists():
        paths.add(frozen_path)
    return sorted(paths)


def _load_result(path: Path) -> dict | None:
    """Load and validate a single result JSON.

    Returns the parsed dict if valid, or None if malformed/error-state.
    Prints warnings for skipped files.
    """
    try:
        with open(path) as f:
            content = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"  WARNING: {path} contains malformed JSON: {exc}", file=sys.stderr)
        return None

    # Check for error-state files (written when a rank failed during training)
    if "error" in content:
        rank = content.get("rank", "?")
        print(
            f"  WARNING: rank {rank} ({path}) failed during training: "
            f"{content['error']}",
            file=sys.stderr,
        )
        return None

    # Validate required keys
    required_keys = {col[0] for col in COLUMNS}
    missing = required_keys - content.keys()
    if missing:
        print(
            f"  WARNING: {path} missing required keys: {missing}",
            file=sys.stderr,
        )
        return None

    return content


def _print_table(rows: list[dict]) -> None:
    """Print a formatted results table to stdout."""
    headers = [col[1] for col in COLUMNS]
    formats = [col[2] for col in COLUMNS]

    # Header widths: max of header length and format width
    widths = []
    for header, fmt in zip(headers, formats):
        # Estimate formatted width from a sample value
        sample_width = len(fmt.format(99999.9999)) if "{" in fmt else len(header)
        widths.append(max(len(header), sample_width))

    # Print header
    header_line = " | ".join(h.rjust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row in rows:
        cells = []
        for (key, _, fmt), width in zip(COLUMNS, widths):
            val = row[key]
            formatted = fmt.format(val)
            cells.append(formatted.rjust(width))
        print(" | ".join(cells))


def _write_csv(rows: list[dict], output_path: Path) -> None:
    """Write results to CSV with the specified column order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [col[1] for col in COLUMNS]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            csv_row = {col[1]: row[col[0]] for col in COLUMNS}
            writer.writerow(csv_row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate LoRA ablation results into a summary table."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="results/ablation_r*.json",
        help='Glob pattern for result files (default: "results/ablation_r*.json").',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/ablation_summary.csv",
        help="Output CSV path (default: results/ablation_summary.csv).",
    )
    args = parser.parse_args()

    # Find result files
    files = _find_result_files(args.pattern)

    if not files:
        print("No ablation results found in results/")
        print("Expected files:")
        for rank in sorted(EXPECTED_RANKS):
            if rank == 0:
                print(f"  results/ablation_frozen.json  (frozen baseline)")
            else:
                print(f"  results/ablation_r{rank}.json")
        print("\nRun the ablation first: python scripts/run_lora_ablation.py")
        sys.exit(0)

    print(f"Found {len(files)} result file(s):\n")

    # Load and validate
    rows = []
    for path in files:
        result = _load_result(path)
        if result is not None:
            rows.append(result)

    if not rows:
        print("No valid result files found.", file=sys.stderr)
        sys.exit(1)

    # Sort by rank ascending (frozen baseline rank=0 sorts first)
    rows.sort(key=lambda r: r["rank"])

    # Print table
    print()
    _print_table(rows)
    print()

    # Report missing ranks
    found_ranks = {row["rank"] for row in rows}
    missing_ranks = EXPECTED_RANKS - found_ranks
    if missing_ranks:
        print(f"Missing ranks: {sorted(missing_ranks)}")
        for rank in sorted(missing_ranks):
            if rank == 0:
                print(f"  rank 0 (frozen baseline): results/ablation_frozen.json")
            else:
                print(f"  rank {rank}: results/ablation_r{rank}.json")
        print()

    # Write CSV
    output_path = Path(args.output)
    _write_csv(rows, output_path)
    print(f"CSV written to {output_path}")


if __name__ == "__main__":
    main()
