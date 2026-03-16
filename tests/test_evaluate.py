"""Tests for evaluation metrics module.

Verifies that compute_metrics produces correct, JSON-serializable output
for binary classification, including edge cases (perfect predictions,
all-wrong, single class).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.evaluate import compute_metrics


class TestComputeMetricsBasic:
    """Core correctness of accuracy, F1, and confusion matrix."""

    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0
        assert metrics["confusion_matrix"] == [[2, 0], [0, 3]]

    def test_all_wrong(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0
        assert metrics["f1_macro"] == 0.0
        assert metrics["confusion_matrix"] == [[0, 2], [2, 0]]

    def test_mixed_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        metrics = compute_metrics(y_true, y_pred)
        # 4 correct out of 6
        assert metrics["accuracy"] == pytest.approx(4 / 6)
        # Confusion: TP(class0)=2, FP(class0)=1, FN(class0)=1
        #            TP(class1)=2, FP(class1)=1, FN(class1)=1
        assert metrics["confusion_matrix"] == [[2, 1], [1, 2]]


class TestComputeMetricsPerClass:
    """Per-class precision/recall/F1 breakdown."""

    def test_per_class_keys_present(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert "not_stressed" in metrics["per_class"]
        assert "stressed" in metrics["per_class"]

    def test_per_class_values_sum_to_total(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0])
        metrics = compute_metrics(y_true, y_pred)
        total_support = sum(
            cls["support"] for cls in metrics["per_class"].values()
        )
        assert total_support == len(y_true)

    def test_custom_label_names(self) -> None:
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        metrics = compute_metrics(y_true, y_pred, label_names=("neg", "pos"))
        assert "neg" in metrics["per_class"]
        assert "pos" in metrics["per_class"]


class TestComputeMetricsValidation:
    """Input validation edge cases."""

    def test_empty_arrays_raise(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_metrics(np.array([]), np.array([]))

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_metrics(np.array([0, 1]), np.array([0]))

    def test_accepts_python_lists(self) -> None:
        """Should work with plain lists (converted via np.asarray internally)."""
        metrics = compute_metrics(
            np.asarray([0, 1, 0, 1]),
            np.asarray([0, 1, 1, 0]),
        )
        assert "accuracy" in metrics


class TestJsonSerializable:
    """Output must be JSON-serializable for results files and W&B."""

    def test_json_round_trip(self) -> None:
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        metrics = compute_metrics(y_true, y_pred)
        # This would raise TypeError if any numpy types leaked through
        serialized = json.dumps(metrics)
        deserialized = json.loads(serialized)
        assert deserialized["accuracy"] == pytest.approx(metrics["accuracy"])

    def test_no_numpy_types_in_output(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert isinstance(metrics["accuracy"], float)
        assert isinstance(metrics["f1_macro"], float)
        assert isinstance(metrics["confusion_matrix"][0][0], int)
