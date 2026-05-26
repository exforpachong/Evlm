#!/usr/bin/env python3
"""
Generate lightweight publication artifacts from existing local results.

This script does not run model inference. It creates:
- data/test_fallback_audit.csv
- docs/publication_readiness_summary.json
- docs/publication_tables.md
- ablation_eval_results_v3/ablation_manifest.json
"""

import csv
import json
from math import comb
from collections import Counter
from datetime import date
from pathlib import Path

from eval_metrics import CANONICAL_LABELS, CORE_LABELS, compute_classification_metrics, strict_correct
from fallback_audit_utils import apply_manual_fallback_labels, load_fallback_review_summary


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
ABLATION_DIR = ROOT / "ablation_eval_results_v3"
FULL_305_DIR = ROOT / "full_305_v3_results"
EVAL_V3_DIR = ROOT / "eval_results_v3"
INTERNVL3_DIR = ROOT / "internvl3_eval_results"
PROMPT_ROBUSTNESS_DIR = ROOT / "prompt_schema_robustness_results"
BASELINE_DIR = ROOT / "baseline_results"


def retrained_ablation_dirs() -> list[Path]:
    dirs = [
        path
        for path in ROOT.glob("ablation_eval_results_v3_retrained_*")
        if path.is_dir()
    ]
    return sorted(dirs, key=lambda path: path.stat().st_mtime, reverse=True)

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metric_total(metrics: dict) -> int:
    return int(metrics.get("total", metrics.get("total_samples", 0)) or 0)


def metric_strict_correct(metrics: dict) -> int:
    return int(metrics.get("strict_correct", metrics.get("strict_correct_count", 0)) or 0)


def fmt_pct(value) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.2f}"


def fmt_float(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def metric_table_row(scope: str, model: str, metrics: dict, note: str = "") -> list[str]:
    total = metric_total(metrics)
    valid_subset_count = int(metrics.get("valid_subset_count", 0) or 0)
    return [
        scope,
        model,
        str(total),
        f"{metric_strict_correct(metrics)}/{total}" if total else "-",
        fmt_pct(metrics.get("accuracy", metrics.get("strict_accuracy"))),
        fmt_float(metrics.get("macro_f1")),
        fmt_float(metrics.get("weighted_f1")),
        fmt_pct(metrics.get("valid_json_rate")),
        str(valid_subset_count),
        fmt_pct(metrics.get("valid_subset_accuracy")),
        fmt_float(metrics.get("valid_subset_macro_f1")),
        fmt_float(metrics.get("valid_subset_weighted_f1")),
        note,
    ]


def compute_metrics(results, labels=CANONICAL_LABELS) -> dict:
    return compute_classification_metrics(results, labels)


def mcnemar_exact_p(zero_only: int, tuned_only: int) -> float | None:
    discordant = zero_only + tuned_only
    if discordant == 0:
        return None
    lower_tail = sum(comb(discordant, k) for k in range(min(zero_only, tuned_only) + 1)) / (2 ** discordant)
    return min(1.0, 2 * lower_tail)


def paired_summary(zero_rows, tuned_rows, labels=CANONICAL_LABELS, source=None) -> dict:
    label_set = set(labels)
    if source:
        zero_rows = [r for r in zero_rows if r.get("label_source") == source]
        tuned_rows = [r for r in tuned_rows if r.get("label_source") == source]

    zero_by_id = {r["id"]: r for r in zero_rows}
    tuned_by_id = {r["id"]: r for r in tuned_rows}
    common_ids = [
        item_id
        for item_id in sorted(set(zero_by_id) & set(tuned_by_id))
        if zero_by_id[item_id].get("ground_truth") in label_set
        and tuned_by_id[item_id].get("ground_truth") in label_set
    ]

    both_correct = 0
    both_wrong = 0
    zero_only = 0
    tuned_only = 0
    for item_id in common_ids:
        zero_is_correct = strict_correct(zero_by_id[item_id], labels)
        tuned_is_correct = strict_correct(tuned_by_id[item_id], labels)
        if zero_is_correct and tuned_is_correct:
            both_correct += 1
        elif zero_is_correct and not tuned_is_correct:
            zero_only += 1
        elif tuned_is_correct and not zero_is_correct:
            tuned_only += 1
        else:
            both_wrong += 1

    total = len(common_ids)
    zero_correct = both_correct + zero_only
    tuned_correct = both_correct + tuned_only
    return {
        "total": total,
        "zero_shot_correct": zero_correct,
        "fine_tuned_correct": tuned_correct,
        "accuracy_diff": (tuned_correct - zero_correct) / total if total else 0.0,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "zero_shot_only_correct": zero_only,
        "fine_tuned_only_correct": tuned_only,
        "mcnemar_exact_p": mcnemar_exact_p(zero_only, tuned_only),
    }


def summarize_full_305():
    zero = load_json(FULL_305_DIR / "zeroshot_results.json")
    tuned = load_json(FULL_305_DIR / "finetuned_results.json")
    fallback_review = load_fallback_review_summary(ROOT)
    use_manual_fallback = bool(fallback_review.get("complete"))

    metric_zero = (
        apply_manual_fallback_labels(zero, fallback_review["manual_labels_by_id"])
        if use_manual_fallback
        else zero
    )
    metric_tuned = (
        apply_manual_fallback_labels(tuned, fallback_review["manual_labels_by_id"])
        if use_manual_fallback
        else tuned
    )

    def by_source(rows, source):
        return [r for r in rows if r.get("label_source") == source]

    summary = {
        "label_policy": "human_reviewed_fallback" if use_manual_fallback else "filename_fallback_weak",
        "fallback_review": {
            key: value
            for key, value in fallback_review.items()
            if key != "manual_labels_by_id"
        },
        "full_305": {
            "zeroshot": compute_metrics(metric_zero),
            "finetuned": compute_metrics(metric_tuned),
        },
        "canonical_subset": {
            "zeroshot": compute_metrics(by_source(metric_zero, "canonical")),
            "finetuned": compute_metrics(by_source(metric_tuned, "canonical")),
        },
        "filename_fallback_subset": {
            "zeroshot": compute_metrics(by_source(metric_zero, "filename_fallback")),
            "finetuned": compute_metrics(by_source(metric_tuned, "filename_fallback")),
        },
        "full_305_5core": {
            "zeroshot": compute_metrics(metric_zero, CORE_LABELS),
            "finetuned": compute_metrics(metric_tuned, CORE_LABELS),
        },
        "canonical_5core": {
            "zeroshot": compute_metrics(by_source(metric_zero, "canonical"), CORE_LABELS),
            "finetuned": compute_metrics(by_source(metric_tuned, "canonical"), CORE_LABELS),
        },
        "weak_label_reference": {
            "full_305": {
                "zeroshot": compute_metrics(zero),
                "finetuned": compute_metrics(tuned),
            },
            "filename_fallback_subset": {
                "zeroshot": compute_metrics(by_source(zero, "filename_fallback")),
                "finetuned": compute_metrics(by_source(tuned, "filename_fallback")),
            },
        },
    }

    summary["paired_significance"] = {
        "full_305": paired_summary(metric_zero, metric_tuned),
        "canonical_subset": paired_summary(metric_zero, metric_tuned, source="canonical"),
        "filename_fallback_subset": paired_summary(metric_zero, metric_tuned, source="filename_fallback"),
        "full_305_5core": paired_summary(metric_zero, metric_tuned, labels=CORE_LABELS),
        "canonical_5core": paired_summary(metric_zero, metric_tuned, labels=CORE_LABELS, source="canonical"),
    }
    return summary


def generate_fallback_audit():
    data = load_json(DATA_DIR / "dataset_test_classification_full.json")
    zero = {r["id"]: r for r in load_json(FULL_305_DIR / "zeroshot_results.json")}
    tuned = {r["id"]: r for r in load_json(FULL_305_DIR / "finetuned_results.json")}
    out_path = DATA_DIR / "test_fallback_audit.csv"
    existing_reviews = {}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                existing_reviews[row.get("id", "")] = {
                    "manual_label": row.get("manual_label", ""),
                    "reviewer_notes": row.get("reviewer_notes", ""),
                }

    rows = []
    for item in data:
        classification = item.get("classification", {})
        if classification.get("label_source") != "filename_fallback":
            continue

        item_id = item.get("id", "")
        weak_label = classification.get("disaster_type", "")
        image = item.get("images", [""])[0]
        zero_pred = zero.get(item_id, {}).get("prediction")
        tuned_pred = tuned.get(item_id, {}).get("prediction")

        zero_matches = zero_pred == weak_label
        tuned_matches = tuned_pred == weak_label
        models_agree = zero_pred == tuned_pred

        if not models_agree or zero_pred == "other" or tuned_pred == "other":
            priority = "high"
        elif zero_matches and tuned_matches:
            priority = "low"
        elif zero_matches or tuned_matches:
            priority = "medium"
        else:
            priority = "high"

        existing = existing_reviews.get(item_id, {})

        rows.append({
            "id": item_id,
            "image": image,
            "weak_label": weak_label,
            "matched_prefix": classification.get("matched_prefix", ""),
            "zeroshot_pred": zero_pred or "",
            "finetuned_pred": tuned_pred or "",
            "zeroshot_matches_weak": zero_matches,
            "finetuned_matches_weak": tuned_matches,
            "models_agree": models_agree,
            "review_priority": priority,
            "manual_label": existing.get("manual_label", ""),
            "reviewer_notes": existing.get("reviewer_notes", ""),
        })

    priority_order = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda r: (priority_order[r["review_priority"]], r["weak_label"], r["id"]))

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    review_summary = load_fallback_review_summary(ROOT)
    public_review_summary = {
        key: value
        for key, value in review_summary.items()
        if key != "manual_labels_by_id"
    }
    public_review_summary.update({
        "path": str(out_path.relative_to(ROOT)),
        "total": len(rows),
        "priority_counts": dict(Counter(r["review_priority"] for r in rows)),
        "weak_label_counts": dict(Counter(r["weak_label"] for r in rows)),
    })
    return public_review_summary


def summarize_ablation():
    manifest = {}
    for scale in ["25", "50", "100"]:
        source_dir = ABLATION_DIR
        for candidate_dir in retrained_ablation_dirs():
            candidate_metrics = candidate_dir / f"scale_{scale}pct.json"
            candidate_results = candidate_dir / f"scale_{scale}pct_results.json"
            if candidate_metrics.exists() and candidate_results.exists():
                source_dir = candidate_dir
                break

        metrics_path = source_dir / f"scale_{scale}pct.json"
        results_path = source_dir / f"scale_{scale}pct_results.json"
        entry = {
            "metrics_file": str(metrics_path.relative_to(ROOT)) if metrics_path.exists() else None,
            "results_file": str(results_path.relative_to(ROOT)) if results_path.exists() else None,
            "source_dir": str(source_dir.relative_to(ROOT)),
            "raw_results_available": results_path.exists(),
            "metrics": load_json(metrics_path) if metrics_path.exists() else None,
        }
        if results_path.exists():
            raw = load_json(results_path)
            entry["raw_result_count"] = len(raw) if isinstance(raw, list) else None
            if isinstance(raw, list) and raw:
                recomputed = compute_metrics(raw)
                entry["recomputed_from_raw"] = recomputed
                reported_accuracy = (entry.get("metrics") or {}).get("accuracy")
                entry["raw_results_consistent_with_metrics"] = (
                    reported_accuracy is not None
                    and abs(recomputed["accuracy"] - reported_accuracy) <= 0.005
                )
                entry["raw_results_usable_for_publication"] = (
                    entry["raw_results_consistent_with_metrics"]
                    and recomputed.get("valid_json_rate", 0.0) >= 0.95
                )
        manifest[f"scale_{scale}pct"] = entry

    out_path = ABLATION_DIR / "ablation_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def load_main_eval_metrics():
    metrics = {}
    for name in ["zeroshot", "finetuned"]:
        results_path = EVAL_V3_DIR / f"{name}_results.json"
        metrics_path = EVAL_V3_DIR / f"{name}_metrics.json"
        if results_path.exists():
            entry = compute_metrics(load_json(results_path))
            if metrics_path.exists():
                entry["reported_metrics_file"] = str(metrics_path.relative_to(ROOT))
            metrics[name] = entry
    return metrics


def load_open_vlm_baselines():
    baselines = {}
    candidates = {
        "internvl3_zeroshot": {
            "label": "InternVL3-2B zero-shot",
            "result_dir": INTERNVL3_DIR,
            "prefix": "zeroshot",
            "note": "Additional open VLM baseline",
        }
    }
    for key, spec in candidates.items():
        result_dir = spec["result_dir"]
        prefix = spec["prefix"]
        results_path = result_dir / f"{prefix}_results.json"
        metrics_path = result_dir / f"{prefix}_metrics.json"
        if not results_path.exists():
            continue
        entry = {
            "label": spec["label"],
            "note": spec["note"],
            "results_file": str(results_path.relative_to(ROOT)),
            "metrics": compute_metrics(load_json(results_path)),
        }
        if metrics_path.exists():
            entry["metrics_file"] = str(metrics_path.relative_to(ROOT))
            entry["reported_metrics"] = load_json(metrics_path)
        baselines[key] = entry
    return baselines


def load_non_vlm_baselines():
    baselines = {}
    candidates = {
        "classical_image_ridge": {
            "label": "Classical image-only ridge classifier",
            "metrics_path": BASELINE_DIR / "classical_image_baseline_metrics.json",
            "results_path": BASELINE_DIR / "classical_image_baseline_results.json",
            "note": "Image-only non-VLM baseline; JSON validity not applicable",
        }
    }
    for key, spec in candidates.items():
        metrics_path = spec["metrics_path"]
        results_path = spec["results_path"]
        if not metrics_path.exists():
            continue
        baselines[key] = {
            "label": spec["label"],
            "note": spec["note"],
            "metrics_file": str(metrics_path.relative_to(ROOT)),
            "results_file": str(results_path.relative_to(ROOT)) if results_path.exists() else None,
            "metrics": load_json(metrics_path),
        }
    return baselines


def load_prompt_schema_robustness():
    summary_path = PROMPT_ROBUSTNESS_DIR / "summary.json"
    if not summary_path.exists():
        return {"exists": False, "path": str(summary_path.relative_to(ROOT))}
    payload = load_json(summary_path)
    payload["exists"] = True
    payload["path"] = str(summary_path.relative_to(ROOT))
    return payload


def write_readiness_summary(summary):
    DOCS_DIR.mkdir(exist_ok=True)

    json_path = DOCS_DIR / "publication_readiness_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    tables_path = DOCS_DIR / "publication_tables.md"
    with open(tables_path, "w", encoding="utf-8") as f:
        f.write(render_publication_tables(summary))

    return {
        "json": str(json_path.relative_to(ROOT)),
        "tables": str(tables_path.relative_to(ROOT)),
    }


def render_publication_tables(summary: dict) -> str:
    generated_on = summary.get("generated_on", "")
    headers = [
        "Scope",
        "Model",
        "N",
        "Strict correct",
        "Strict acc (%)",
        "Strict macro F1",
        "Strict weighted F1",
        "Valid JSON (%)",
        "Valid-subset N",
        "Valid-subset acc (%)",
        "Valid-subset macro F1",
        "Valid-subset weighted F1",
        "Note",
    ]

    main_eval = summary["main_eval_v3"]
    full_305 = summary["full_305_v3"]
    uses_human_fallback = full_305.get("label_policy") == "human_reviewed_fallback"
    full_note = "Includes human-reviewed fallback labels" if uses_human_fallback else "Robustness only"
    fallback_note = "Human-reviewed fallback labels" if uses_human_fallback else "Weak labels need audit"
    rows = [
        metric_table_row("Canonical clean test", "Qwen3-VL-2B zero-shot", main_eval["zeroshot"], "Primary benchmark"),
        metric_table_row("Canonical clean test", "Qwen3-VL-2B LoRA", main_eval["finetuned"], "Primary benchmark"),
    ]
    for baseline in summary.get("non_vlm_baselines", {}).values():
        rows.append(
            metric_table_row(
                "Canonical clean test",
                baseline.get("label", "Non-VLM baseline"),
                baseline.get("metrics", {}),
                baseline.get("note", "Image-only non-VLM baseline"),
            )
        )
    for baseline in summary.get("open_vlm_baselines", {}).values():
        rows.append(
            metric_table_row(
                "Canonical clean test",
                baseline.get("label", "Open VLM baseline"),
                baseline.get("metrics", {}),
                baseline.get("note", "Additional open VLM baseline"),
            )
        )
    rows.extend([
        metric_table_row("Full 305", "Qwen3-VL-2B zero-shot", full_305["full_305"]["zeroshot"], full_note),
        metric_table_row("Full 305", "Qwen3-VL-2B LoRA", full_305["full_305"]["finetuned"], full_note),
        metric_table_row("Full 305 canonical subset", "Qwen3-VL-2B zero-shot", full_305["canonical_subset"]["zeroshot"], "Matches canonical clean"),
        metric_table_row("Full 305 canonical subset", "Qwen3-VL-2B LoRA", full_305["canonical_subset"]["finetuned"], "Matches canonical clean"),
        metric_table_row("Full 305 filename fallback subset", "Qwen3-VL-2B zero-shot", full_305["filename_fallback_subset"]["zeroshot"], fallback_note),
        metric_table_row("Full 305 filename fallback subset", "Qwen3-VL-2B LoRA", full_305["filename_fallback_subset"]["finetuned"], fallback_note),
        metric_table_row("Canonical clean 5-core", "Qwen3-VL-2B zero-shot", full_305["canonical_5core"]["zeroshot"], "Excludes other"),
        metric_table_row("Canonical clean 5-core", "Qwen3-VL-2B LoRA", full_305["canonical_5core"]["finetuned"], "Excludes other"),
        metric_table_row("Full 305 5-core", "Qwen3-VL-2B zero-shot", full_305["full_305_5core"]["zeroshot"], "Excludes other"),
        metric_table_row("Full 305 5-core", "Qwen3-VL-2B LoRA", full_305["full_305_5core"]["finetuned"], "Excludes other"),
    ])

    ablation_rows = []
    for scale in ["25", "50", "100"]:
        key = f"scale_{scale}pct"
        entry = summary["ablation_manifest"].get(key, {})
        metrics = entry.get("recomputed_from_raw") or entry.get("metrics") or {}
        usable = entry.get("raw_results_usable_for_publication")
        if usable:
            note = "Usable"
        else:
            note = "Do not report as final; retrain/rerun required"
        ablation_rows.append(metric_table_row(f"{scale}% train data", "Qwen3-VL-2B LoRA", metrics, note))

    sig_headers = [
        "Scope",
        "N",
        "Zero-shot correct",
        "LoRA correct",
        "Delta acc (%)",
        "Zero-shot only",
        "LoRA only",
        "McNemar exact p",
    ]
    sig_rows = []
    for scope, key in [
        ("Full 305", "full_305"),
        ("Full 305 canonical subset", "canonical_subset"),
        ("Full 305 filename fallback subset", "filename_fallback_subset"),
        ("Full 305 5-core", "full_305_5core"),
        ("Canonical 5-core", "canonical_5core"),
    ]:
        sig = full_305["paired_significance"][key]
        p_value = sig.get("mcnemar_exact_p")
        sig_rows.append([
            scope,
            str(sig.get("total", 0)),
            str(sig.get("zero_shot_correct", 0)),
            str(sig.get("fine_tuned_correct", 0)),
            fmt_pct(sig.get("accuracy_diff")),
            str(sig.get("zero_shot_only_correct", 0)),
            str(sig.get("fine_tuned_only_correct", 0)),
            "-" if p_value is None else f"{float(p_value):.4g}",
        ])

    fallback = summary.get("fallback_audit", {})
    fallback_counts = fallback.get("priority_counts", {})
    pending_fallback = int(fallback.get("pending_manual_review", fallback.get("total", 0)) or 0)
    if pending_fallback:
        fallback_status_line = (
            f"- Fallback samples awaiting manual review: {pending_fallback}; "
            f"priority counts: {fallback_counts}."
        )
    else:
        fallback_status_line = (
            f"- Fallback manual review complete: {fallback.get('manual_label_filled', 0)}/"
            f"{fallback.get('rows', fallback.get('total', 0))} labels filled; "
            f"changed from weak labels: {fallback.get('changed_from_weak_labels', 0)}."
        )
    unusable_scales = [
        key.replace("scale_", "").replace("pct", "%")
        for key, entry in summary.get("ablation_manifest", {}).items()
        if not entry.get("raw_results_usable_for_publication")
    ]
    if unusable_scales:
        ablation_status_line = (
            "- Data-scale ablation scales requiring retrain/rerun: "
            + ", ".join(unusable_scales)
            + "."
        )
    else:
        ablation_status_line = "- Data-scale ablation raw results are publication-ready under the current validity checks."

    prompt_headers = [
        "Prompt",
        "Mode",
        "N",
        "Strict correct",
        "Strict acc (%)",
        "Macro F1",
        "Valid JSON (%)",
        "Delta acc vs current (pp)",
        "Delta valid JSON vs current (pp)",
    ]
    prompt_rows = []
    prompt_summary = summary.get("prompt_schema_robustness", {})
    for row in prompt_summary.get("rows", []):
        delta_acc = row.get("delta_strict_accuracy_vs_current_zh")
        delta_json = row.get("delta_valid_json_rate_vs_current_zh")
        prompt_rows.append([
            row.get("prompt_variant", ""),
            row.get("mode", ""),
            str(row.get("total", 0)),
            f"{row.get('strict_correct', 0)}/{row.get('total', 0)}",
            fmt_pct(row.get("strict_accuracy")),
            fmt_float(row.get("macro_f1")),
            fmt_pct(row.get("valid_json_rate")),
            "-" if delta_acc is None else f"{float(delta_acc) * 100:+.2f}",
            "-" if delta_json is None else f"{float(delta_json) * 100:+.2f}",
        ])

    prompt_section = []
    if prompt_rows:
        prompt_section = [
            "## Prompt/Schema Robustness",
            "",
            markdown_table(prompt_headers, prompt_rows),
            "",
            "Use as protocol robustness evidence, not as a new model-performance claim.",
            "",
        ]
    lines = [
        "# Publication Tables",
        "",
        f"Generated on: {generated_on}",
        "",
        "These tables report strict metrics and valid-subset metrics side by side. Strict metrics keep every eligible sample in the denominator and count invalid JSON, missing predictions, and non-canonical predictions as wrong. Valid-subset metrics use only samples with canonical predictions.",
        "",
        "## Main and Robustness Results",
        "",
        markdown_table(headers, rows),
        "",
        "## Data-Scale Ablation Status",
        "",
        markdown_table(headers, ablation_rows),
        "",
        *prompt_section,
        "## Paired Significance",
        "",
        markdown_table(sig_headers, sig_rows),
        "",
        "## Remaining P0 Checks",
        "",
        f"- Fallback audit file: `{fallback.get('path', 'data/test_fallback_audit.csv')}`.",
        fallback_status_line,
        ablation_status_line,
        "",
    ]
    return "\n".join(lines)


def main():
    fallback_audit = generate_fallback_audit()
    summary = {
        "generated_on": date.today().isoformat(),
        "main_eval_v3": load_main_eval_metrics(),
        "non_vlm_baselines": load_non_vlm_baselines(),
        "open_vlm_baselines": load_open_vlm_baselines(),
        "full_305_v3": summarize_full_305(),
        "fallback_audit": fallback_audit,
        "ablation_manifest": summarize_ablation(),
        "prompt_schema_robustness": load_prompt_schema_robustness(),
    }
    written = write_readiness_summary(summary)
    print("Generated publication artifacts:")
    print(f"  {summary['fallback_audit']['path']}")
    print(f"  {written['json']}")
    print(f"  {written['tables']}")
    print("  ablation_eval_results_v3/ablation_manifest.json")


if __name__ == "__main__":
    main()
