#!/usr/bin/env python3
"""Recompute experiment metrics from raw result files and flag inconsistencies.

This script does not run model inference. It is intended as a publication
readiness check before copying benchmark numbers into reports.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from eval_metrics import CANONICAL_LABELS, CORE_LABELS, compute_classification_metrics
from fallback_audit_utils import apply_manual_fallback_labels, load_fallback_review_summary


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"


def retrained_ablation_dirs() -> list[Path]:
    dirs = [
        path
        for path in ROOT.glob("ablation_eval_results_v3_retrained_*")
        if path.is_dir()
    ]
    return sorted(dirs, key=lambda path: path.stat().st_mtime, reverse=True)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_row(row: dict) -> dict:
    normalized = dict(row)
    if "ground_truth" not in normalized and "gt" in normalized:
        normalized["ground_truth"] = normalized.get("gt")
    if "prediction" not in normalized and "pred" in normalized:
        normalized["prediction"] = normalized.get("pred")
    return normalized


def summarize_result_file(path: Path, labels: list[str] | None = None) -> dict:
    labels = labels or CANONICAL_LABELS
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "exists": False}
    rows = load_json(path) if path.suffix == ".json" else load_jsonl(path)
    rows = [normalize_row(row) for row in rows]
    metrics = compute_classification_metrics(rows, labels)
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": True,
        "raw_count": len(rows),
        "metrics": metrics,
    }


def compare_reported_metrics(metrics_path: Path, recomputed: dict) -> dict:
    if not metrics_path.exists():
        return {
            "metrics_path": str(metrics_path.relative_to(ROOT)),
            "exists": False,
            "consistent_with_raw": None,
        }

    reported = load_json(metrics_path)
    comparisons = {}
    for key in ["accuracy", "strict_accuracy", "macro_f1", "weighted_f1", "valid_json_rate"]:
        if key in reported and key in recomputed:
            comparisons[key] = {
                "reported": reported[key],
                "recomputed": recomputed[key],
                "delta": recomputed[key] - reported[key],
                "within_0_005": abs(recomputed[key] - reported[key]) <= 0.005,
            }

    comparable = [item["within_0_005"] for item in comparisons.values()]
    return {
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "exists": True,
        "comparisons": comparisons,
        "consistent_with_raw": all(comparable) if comparable else None,
    }


def summarize_eval_directory(result_dir: Path, result_prefixes: list[str]) -> dict:
    summary = {}
    for prefix in result_prefixes:
        result_path = result_dir / f"{prefix}_results.json"
        metrics_path = result_dir / f"{prefix}_metrics.json"
        result_summary = summarize_result_file(result_path)
        if result_summary.get("exists"):
            result_summary["reported_metrics_check"] = compare_reported_metrics(
                metrics_path,
                result_summary["metrics"],
            )
            result_summary["core_5class_metrics"] = compute_classification_metrics(
                [normalize_row(row) for row in load_json(result_path)],
                CORE_LABELS,
            )
        summary[prefix] = result_summary
    return summary


def summarize_ablation() -> dict:
    summary = {}
    for scale in ["25", "50", "100"]:
        result_dir = ROOT / "ablation_eval_results_v3"
        for candidate_dir in retrained_ablation_dirs():
            if (
                (candidate_dir / f"scale_{scale}pct_results.json").exists()
                and (candidate_dir / f"scale_{scale}pct.json").exists()
            ):
                result_dir = candidate_dir
                break

        result_path = result_dir / f"scale_{scale}pct_results.json"
        metrics_path = result_dir / f"scale_{scale}pct.json"
        result_summary = summarize_result_file(result_path)
        result_summary["source_dir"] = str(result_dir.relative_to(ROOT))
        if result_summary.get("exists"):
            result_summary["reported_metrics_check"] = compare_reported_metrics(
                metrics_path,
                result_summary["metrics"],
            )
        elif metrics_path.exists():
            result_summary["reported_metrics_only"] = load_json(metrics_path)
        summary[f"scale_{scale}pct"] = result_summary
    return summary


def summarize_glm() -> dict:
    result_dir = ROOT / "glm_eval_results"
    summary = {}
    for name in ["results", "results_constrained"]:
        path = result_dir / f"{name}.jsonl"
        result_summary = summarize_result_file(path)
        if path.name == "results.jsonl" and result_summary.get("exists"):
            result_summary["reported_metrics_check"] = compare_reported_metrics(
                result_dir / "metrics.json",
                result_summary["metrics"],
            )
        summary[name] = result_summary
    normalized_path = result_dir / "normalized_metrics.json"
    if normalized_path.exists():
        summary["normalized_metrics"] = load_json(normalized_path)
    return summary


def summarize_open_vlm_baselines() -> dict:
    return {
        "internvl3_zeroshot": summarize_result_file(
            ROOT / "internvl3_eval_results" / "zeroshot_results.json"
        )
    }


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_image_overlaps(files: list[Path]) -> dict:
    by_hash = defaultdict(list)
    missing = []
    for file_path in files:
        split = file_path.name.split("_")[1]
        for item in load_json(file_path):
            images = item.get("images") or []
            if not images:
                missing.append({"split": split, "id": item.get("id"), "image": None})
                continue
            image_path = ROOT / "sample_images" / Path(images[0]).name
            if not image_path.exists():
                missing.append({"split": split, "id": item.get("id"), "image": str(image_path)})
                continue
            by_hash[md5_file(image_path)].append({
                "split": split,
                "id": item.get("id"),
                "image": str(image_path.relative_to(ROOT)),
            })

    overlaps = [
        {"md5": digest, "samples": samples}
        for digest, samples in by_hash.items()
        if len({sample["split"] for sample in samples}) > 1
    ]
    split_pair_counts = Counter()
    test_involved = []
    for overlap in overlaps:
        splits = sorted({sample["split"] for sample in overlap["samples"]})
        split_pair_counts["+".join(splits)] += 1
        if "test" in splits:
            test_involved.append(overlap)

    return {
        "missing_images": missing,
        "cross_split_md5_overlap_count": len(overlaps),
        "split_pair_counts": dict(split_pair_counts),
        "test_involved_overlap_count": len(test_involved),
        "test_involved_overlap_examples": test_involved[:10],
        "cross_split_md5_overlap_examples": overlaps[:10],
    }


def summarize_data_integrity() -> dict:
    fallback_summary = load_fallback_review_summary(ROOT)
    fallback_summary.pop("manual_labels_by_id", None)

    canonical_files = [
        DATA_DIR / "dataset_train_canonical_clean.json",
        DATA_DIR / "dataset_val_canonical_clean.json",
        DATA_DIR / "dataset_test_canonical_clean.json",
    ]
    full_files = [
        DATA_DIR / "dataset_train_classification_full.json",
        DATA_DIR / "dataset_val_classification_full.json",
        DATA_DIR / "dataset_test_classification_full.json",
    ]
    return {
        "fallback_audit": fallback_summary,
        "canonical_clean_split_images": split_image_overlaps(canonical_files),
        "classification_full_split_images": split_image_overlaps(full_files),
    }


def summarize_full_305_human_reviewed() -> dict:
    fallback_summary = load_fallback_review_summary(ROOT)
    public_fallback_summary = {
        key: value
        for key, value in fallback_summary.items()
        if key != "manual_labels_by_id"
    }
    summary = {"fallback_review": public_fallback_summary}
    if not fallback_summary.get("complete"):
        summary["exists"] = False
        summary["reason"] = "fallback manual review is incomplete or contains invalid labels"
        return summary

    for prefix in ["zeroshot", "finetuned"]:
        result_path = ROOT / "full_305_v3_results" / f"{prefix}_results.json"
        result_summary = summarize_result_file(result_path)
        if result_summary.get("exists"):
            reviewed_rows = apply_manual_fallback_labels(
                [normalize_row(row) for row in load_json(result_path)],
                fallback_summary["manual_labels_by_id"],
            )
            result_summary["metrics"] = compute_classification_metrics(reviewed_rows)
            result_summary["core_5class_metrics"] = compute_classification_metrics(
                reviewed_rows,
                CORE_LABELS,
            )
            result_summary["label_policy"] = "human_reviewed_fallback"
        summary[prefix] = result_summary
    summary["exists"] = True
    return summary


def main() -> None:
    report = {
        "generated_on": date.today().isoformat(),
        "note": "Metrics are recomputed with strict denominator: invalid JSON and non-canonical predictions count as wrong.",
        "canonical_eval_v3": summarize_eval_directory(ROOT / "eval_results_v3", ["zeroshot", "finetuned"]),
        "full_305_v3": summarize_eval_directory(ROOT / "full_305_v3_results", ["zeroshot", "finetuned"]),
        "full_305_v3_human_reviewed": summarize_full_305_human_reviewed(),
        "open_vlm_baselines": summarize_open_vlm_baselines(),
        "ablation": summarize_ablation(),
        "glm": summarize_glm(),
        "data_integrity": summarize_data_integrity(),
    }

    DOCS_DIR.mkdir(exist_ok=True)
    out_path = DOCS_DIR / "experiment_integrity_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path.relative_to(ROOT)}")
    for section in ["canonical_eval_v3", "full_305_v3", "ablation"]:
        print(f"{section}:")
        for name, item in report[section].items():
            if not item.get("exists"):
                print(f"  {name}: no raw results")
                continue
            metrics = item["metrics"]
            print(
                f"  {name}: acc={metrics['accuracy']:.4f}, "
                f"macro_f1={metrics['macro_f1']:.4f}, "
                f"valid_json={metrics['valid_json_rate']:.2%}"
            )
    print("open_vlm_baselines:")
    for name, item in report["open_vlm_baselines"].items():
        if not item.get("exists"):
            print(f"  {name}: no raw results")
            continue
        metrics = item["metrics"]
        print(
            f"  {name}: acc={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, "
            f"valid_json={metrics['valid_json_rate']:.2%}"
        )
    reviewed = report.get("full_305_v3_human_reviewed", {})
    if reviewed.get("exists"):
        print("full_305_v3_human_reviewed:")
        for name in ["zeroshot", "finetuned"]:
            metrics = reviewed[name]["metrics"]
            print(
                f"  {name}: acc={metrics['accuracy']:.4f}, "
                f"macro_f1={metrics['macro_f1']:.4f}, "
                f"valid_json={metrics['valid_json_rate']:.2%}"
            )


if __name__ == "__main__":
    main()
