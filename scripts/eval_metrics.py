#!/usr/bin/env python3
"""Shared strict classification metrics for disaster benchmark evaluations."""

from __future__ import annotations

from typing import Any, Iterable

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


CANONICAL_LABELS = [
    "flood",
    "earthquake",
    "fire",
    "landslide",
    "windstorm_or_typhoon",
    "other",
]
CORE_LABELS = ["flood", "earthquake", "fire", "landslide", "windstorm_or_typhoon"]
INVALID_PREDICTION_LABEL = "__invalid_or_noncanonical__"


def _get(row: Any, field: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(field, default)
    return getattr(row, field, default)


def get_ground_truth(row: Any) -> Any:
    return _get(row, "ground_truth")


def get_prediction(row: Any) -> Any:
    return _get(row, "prediction")


def is_valid_json(row: Any) -> bool:
    return bool(_get(row, "valid_json", True))


def is_canonical_prediction(row: Any, labels: Iterable[str] = CANONICAL_LABELS) -> bool:
    labels = set(labels)
    return is_valid_json(row) and get_prediction(row) in labels


def strict_correct(row: Any, labels: Iterable[str] = CANONICAL_LABELS) -> bool:
    labels = set(labels)
    return (
        get_ground_truth(row) in labels
        and is_valid_json(row)
        and get_prediction(row) == get_ground_truth(row)
    )


def strict_label_sequences(rows: list[Any], labels: list[str]) -> tuple[list[str], list[str]]:
    y_true = []
    y_pred = []
    for row in rows:
        ground_truth = get_ground_truth(row)
        if ground_truth not in labels:
            continue
        y_true.append(ground_truth)
        if is_canonical_prediction(row, labels):
            y_pred.append(get_prediction(row))
        else:
            y_pred.append(INVALID_PREDICTION_LABEL)
    return y_true, y_pred


def compute_classification_metrics(
    rows: list[Any],
    labels: list[str] | None = None,
    *,
    include_per_class: bool = True,
) -> dict:
    """Compute strict and valid-subset metrics.

    Strict metrics keep every sample with a canonical ground truth in the
    denominator. Invalid JSON, missing predictions, and non-canonical labels are
    counted as wrong predictions.
    """

    labels = labels or CANONICAL_LABELS
    task_rows = [row for row in rows if get_ground_truth(row) in labels]
    total = len(task_rows)
    valid_json_count = sum(1 for row in task_rows if is_valid_json(row))
    canonical_prediction_count = sum(1 for row in task_rows if is_canonical_prediction(row, labels))
    noncanonical_prediction_count = sum(
        1
        for row in task_rows
        if is_valid_json(row)
        and get_prediction(row) is not None
        and get_prediction(row) not in labels
    )
    missing_prediction_count = sum(1 for row in task_rows if get_prediction(row) is None)
    strict_correct_count = sum(1 for row in task_rows if strict_correct(row, labels))

    y_true_strict, y_pred_strict = strict_label_sequences(task_rows, labels)
    if total:
        strict_macro_f1 = float(
            f1_score(y_true_strict, y_pred_strict, labels=labels, average="macro", zero_division=0)
        )
        strict_weighted_f1 = float(
            f1_score(y_true_strict, y_pred_strict, labels=labels, average="weighted", zero_division=0)
        )
        matrix_labels = labels + [INVALID_PREDICTION_LABEL]
        cm = confusion_matrix(y_true_strict, y_pred_strict, labels=matrix_labels).tolist()
    else:
        strict_macro_f1 = 0.0
        strict_weighted_f1 = 0.0
        matrix_labels = labels + [INVALID_PREDICTION_LABEL]
        cm = []

    valid_subset_rows = [row for row in task_rows if is_canonical_prediction(row, labels)]
    if valid_subset_rows:
        valid_y_true = [get_ground_truth(row) for row in valid_subset_rows]
        valid_y_pred = [get_prediction(row) for row in valid_subset_rows]
        valid_subset_correct = sum(1 for gt, pred in zip(valid_y_true, valid_y_pred) if gt == pred)
        valid_subset_accuracy = valid_subset_correct / len(valid_subset_rows)
        valid_subset_macro_f1 = float(
            f1_score(valid_y_true, valid_y_pred, labels=labels, average="macro", zero_division=0)
        )
        valid_subset_weighted_f1 = float(
            f1_score(valid_y_true, valid_y_pred, labels=labels, average="weighted", zero_division=0)
        )
    else:
        valid_subset_correct = 0
        valid_subset_accuracy = 0.0
        valid_subset_macro_f1 = 0.0
        valid_subset_weighted_f1 = 0.0

    per_class = {}
    if include_per_class and total:
        for label in labels:
            true_binary = [1 if y == label else 0 for y in y_true_strict]
            pred_binary = [1 if y == label else 0 for y in y_pred_strict]
            per_class[label] = {
                "precision": float(precision_score(true_binary, pred_binary, zero_division=0)),
                "recall": float(recall_score(true_binary, pred_binary, zero_division=0)),
                "f1": float(f1_score(true_binary, pred_binary, zero_division=0)),
                "support": int(sum(true_binary)),
            }

    accuracy = strict_correct_count / total if total else 0.0
    return {
        "total": total,
        "strict_correct": strict_correct_count,
        "accuracy": accuracy,
        "strict_accuracy": accuracy,
        "macro_f1": strict_macro_f1,
        "weighted_f1": strict_weighted_f1,
        "valid_json_count": valid_json_count,
        "valid_json_rate": valid_json_count / total if total else 0.0,
        "invalid_json_count": total - valid_json_count,
        "canonical_prediction_count": canonical_prediction_count,
        "canonical_prediction_rate": canonical_prediction_count / total if total else 0.0,
        "noncanonical_prediction_count": noncanonical_prediction_count,
        "missing_prediction_count": missing_prediction_count,
        "valid_subset_count": len(valid_subset_rows),
        "valid_subset_correct": valid_subset_correct,
        "valid_subset_accuracy": valid_subset_accuracy,
        "valid_subset_macro_f1": valid_subset_macro_f1,
        "valid_subset_weighted_f1": valid_subset_weighted_f1,
        "per_class_metrics": per_class,
        "confusion_matrix_labels": matrix_labels,
        "confusion_matrix": cm,
    }
