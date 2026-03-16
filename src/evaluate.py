"""
Evaluation metrics for DREADDIT binary classification.

This module is intentionally decoupled from model code and torch — it takes
numpy arrays of true and predicted labels and returns JSON-serializable metric
dicts. This separation means the same evaluation works whether predictions come
from a live training run, a saved checkpoint, or a mock for testing.

The metrics here feed directly into the comparison table from the architecture
plan (frozen baseline / LoRA rank-4 / LoRA rank-16 / Hopfield / full fine-tune).
Accuracy alone is insufficient for a portfolio piece — F1 (macro) normalizes
for any class imbalance, and the confusion matrix reveals the failure mode
(false positives vs. false negatives in stress detection).

Phase 1 connection: Phase 1's RAG evaluation used Hit@K and MRR for retrieval
quality. Phase 2 shifts to classification metrics because we're evaluating the
*attention mechanism itself* (via LoRA fine-tuning), not a retrieval pipeline.
The narrative thread: Phase 1 identified that semantic matching is the bottleneck;
Phase 2 measures how well adapted attention mechanisms resolve that bottleneck
on informal clinical-adjacent text.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: tuple[str, ...] = ("not_stressed", "stressed"),
) -> dict:
    """Compute classification metrics for DREADDIT binary stress detection.

    Returns a flat, JSON-serializable dict suitable for W&B logging (caller's
    responsibility) and results file serialization.

    Args:
        y_true: Ground-truth labels, shape (N,), values in {0, 1}.
        y_pred: Predicted labels, shape (N,), values in {0, 1}.
        label_names: Human-readable class names for the report.

    Returns:
        Dict with keys:
            - accuracy: float
            - f1_macro: float (treats both classes equally)
            - confusion_matrix: list[list[int]] (2x2, row=true, col=pred)
            - per_class: dict mapping class name -> {precision, recall, f1, support}

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError(
            f"Cannot compute metrics on empty arrays: "
            f"len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}."
        )
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, "
            f"y_pred has {len(y_pred)}."
        )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true, y_pred, target_names=list(label_names), output_dict=True
    )

    per_class = {}
    for name in label_names:
        if name in report:
            cls_metrics = report[name]
            per_class[name] = {
                "precision": float(cls_metrics["precision"]),
                "recall": float(cls_metrics["recall"]),
                "f1": float(cls_metrics["f1-score"]),
                "support": int(cls_metrics["support"]),
            }

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
