#!/usr/bin/env python3
"""Helpers for applying human-reviewed fallback labels."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from eval_metrics import CANONICAL_LABELS


FALLBACK_AUDIT_RELATIVE_PATH = Path("data") / "test_fallback_audit.csv"


def load_fallback_review_summary(root: Path) -> dict[str, Any]:
    """Load fallback review labels and summarize review completeness."""

    path = root / FALLBACK_AUDIT_RELATIVE_PATH
    if not path.exists():
        return {
            "exists": False,
            "path": str(FALLBACK_AUDIT_RELATIVE_PATH),
            "rows": 0,
            "manual_label_filled": 0,
            "pending_manual_review": 0,
            "invalid_manual_label_count": 0,
            "manual_labels_by_id": {},
            "manual_label_counts": {},
            "changed_from_weak_labels": 0,
            "complete": False,
        }

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    valid_labels = set(CANONICAL_LABELS)
    labels_by_id: dict[str, str] = {}
    invalid_rows = []
    changed_from_weak = 0

    for row in rows:
        item_id = (row.get("id") or "").strip()
        label = (row.get("manual_label") or "").strip()
        if not label:
            continue
        if label not in valid_labels:
            invalid_rows.append({"id": item_id, "manual_label": label})
            continue
        labels_by_id[item_id] = label
        if label != (row.get("weak_label") or "").strip():
            changed_from_weak += 1

    pending = len(rows) - len(labels_by_id) - len(invalid_rows)
    return {
        "exists": True,
        "path": str(FALLBACK_AUDIT_RELATIVE_PATH),
        "rows": len(rows),
        "manual_label_filled": len(labels_by_id),
        "pending_manual_review": pending,
        "invalid_manual_label_count": len(invalid_rows),
        "invalid_manual_label_examples": invalid_rows[:10],
        "manual_labels_by_id": labels_by_id,
        "manual_label_counts": dict(Counter(labels_by_id.values())),
        "changed_from_weak_labels": changed_from_weak,
        "complete": len(rows) > 0 and pending == 0 and not invalid_rows,
    }


def apply_manual_fallback_labels(
    rows: list[dict[str, Any]],
    manual_labels_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    """Return result rows with filename-fallback ground truth replaced by manual labels."""

    patched_rows = []
    for row in rows:
        patched = dict(row)
        if patched.get("label_source") == "filename_fallback":
            manual_label = manual_labels_by_id.get(str(patched.get("id", "")))
            if manual_label:
                patched["weak_ground_truth"] = patched.get("ground_truth")
                patched["ground_truth"] = manual_label
                patched["ground_truth_source"] = "manual_fallback_review"
                patched["manual_label_applied"] = True
        patched_rows.append(patched)
    return patched_rows
