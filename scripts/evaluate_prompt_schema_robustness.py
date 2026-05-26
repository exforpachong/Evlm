#!/usr/bin/env python3
"""Run prompt/schema robustness checks for the canonical disaster benchmark.

The experiment compares the main Chinese structured prompt against a stricter
English JSON-schema prompt. It reports strict accuracy and valid JSON rate
together, matching the publication protocol.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from evaluate_v3 import EvalConfig, PROMPT_VARIANTS, run_evaluation, save_results


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAIN_RESULTS_DIR = ROOT / "eval_results_v3"


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def copy_main_result(mode: str, target_dir: Path) -> bool:
    """Copy the existing main-prompt result if it is available."""
    source_results = DEFAULT_MAIN_RESULTS_DIR / f"{mode}_results.json"
    source_metrics = DEFAULT_MAIN_RESULTS_DIR / f"{mode}_metrics.json"
    source_config = DEFAULT_MAIN_RESULTS_DIR / f"{mode}_eval_config.json"
    if not source_results.exists() or not source_metrics.exists():
        return False
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_results, target_dir / f"{mode}_results.json")
    shutil.copy2(source_metrics, target_dir / f"{mode}_metrics.json")
    if source_config.exists():
        shutil.copy2(source_config, target_dir / f"{mode}_eval_config.json")
    return True


def metric_total(metrics: dict[str, Any]) -> int:
    return int(metrics.get("total", metrics.get("total_samples", 0)) or 0)


def metric_strict_correct(metrics: dict[str, Any]) -> int:
    return int(metrics.get("strict_correct", metrics.get("strict_correct_count", 0)) or 0)


def metric_accuracy(metrics: dict[str, Any]) -> float:
    return float(metrics.get("accuracy", metrics.get("strict_accuracy", 0.0)) or 0.0)


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}"


def collect_metrics(output_dir: Path, variants: list[str], modes: list[str]) -> list[dict[str, Any]]:
    rows = []
    for variant in variants:
        variant_dir = output_dir / variant
        for mode in modes:
            metrics_path = variant_dir / f"{mode}_metrics.json"
            if not metrics_path.exists():
                continue
            metrics = load_json(metrics_path)
            total = metric_total(metrics)
            rows.append(
                {
                    "prompt_variant": variant,
                    "mode": mode,
                    "total": total,
                    "strict_correct": metric_strict_correct(metrics),
                    "strict_accuracy": metric_accuracy(metrics),
                    "macro_f1": float(metrics.get("macro_f1", 0.0) or 0.0),
                    "weighted_f1": float(metrics.get("weighted_f1", 0.0) or 0.0),
                    "valid_json_rate": float(metrics.get("valid_json_rate", 0.0) or 0.0),
                    "valid_subset_count": int(metrics.get("valid_subset_count", 0) or 0),
                    "valid_subset_accuracy": float(metrics.get("valid_subset_accuracy", 0.0) or 0.0),
                    "source_metrics_path": str(metrics_path.relative_to(ROOT)),
                }
            )
    return rows


def add_prompt_deltas(rows: list[dict[str, Any]]) -> None:
    baseline = {
        row["mode"]: row
        for row in rows
        if row["prompt_variant"] == "current_zh"
    }
    for row in rows:
        base = baseline.get(row["mode"])
        if not base or row["prompt_variant"] == "current_zh":
            row["delta_strict_accuracy_vs_current_zh"] = None
            row["delta_valid_json_rate_vs_current_zh"] = None
            continue
        row["delta_strict_accuracy_vs_current_zh"] = (
            row["strict_accuracy"] - base["strict_accuracy"]
        )
        row["delta_valid_json_rate_vs_current_zh"] = (
            row["valid_json_rate"] - base["valid_json_rate"]
        )


def write_summary(output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    add_prompt_deltas(rows)
    payload = {
        "generated_on": date.today().isoformat(),
        "test_data": args.test_data,
        "limit": args.limit,
        "modes": args.modes,
        "prompt_variants": args.variants,
        "note": (
            "Strict metrics keep every canonical-ground-truth sample in the denominator. "
            "Invalid JSON, missing predictions, and non-canonical predictions count as wrong."
        ),
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    headers = [
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
    lines = [
        "# Prompt/Schema Robustness Summary",
        "",
        f"Generated on: {payload['generated_on']}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        delta_acc = row["delta_strict_accuracy_vs_current_zh"]
        delta_json = row["delta_valid_json_rate_vs_current_zh"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["prompt_variant"],
                    row["mode"],
                    str(row["total"]),
                    f"{row['strict_correct']}/{row['total']}",
                    fmt_pct(row["strict_accuracy"]),
                    f"{row['macro_f1']:.3f}",
                    fmt_pct(row["valid_json_rate"]),
                    "-" if delta_acc is None else f"{delta_acc * 100:+.2f}",
                    "-" if delta_json is None else f"{delta_json * 100:+.2f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Use this table as a protocol-robustness check, not as a new model-performance claim.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_csv_arg(value: str, valid: set[str]) -> list[str]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [part for part in parts if part not in valid]
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid values: {', '.join(invalid)}")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prompt/schema robustness.")
    parser.add_argument("--model_path", default="models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter_path", default="finetune_output_v2/final_adapter")
    parser.add_argument("--test_data", default="data/dataset_test_canonical_clean.json")
    parser.add_argument("--output_dir", default="prompt_schema_robustness_results")
    parser.add_argument(
        "--variants",
        type=lambda value: parse_csv_arg(value, set(PROMPT_VARIANTS)),
        default=["current_zh", "schema_en"],
        help="Comma-separated prompt variants. Available: current_zh,schema_en.",
    )
    parser.add_argument(
        "--modes",
        type=lambda value: parse_csv_arg(value, {"zeroshot", "finetuned"}),
        default=["zeroshot", "finetuned"],
        help="Comma-separated modes. Available: zeroshot,finetuned.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse_current_main_results", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    completed_variants = []
    for variant in args.variants:
        variant_dir = output_dir / variant
        completed_variants.append(variant)
        for mode in args.modes:
            if (
                args.reuse_current_main_results
                and variant == "current_zh"
                and args.limit is None
                and copy_main_result(mode, variant_dir)
            ):
                print(f"Reused main result for {variant}/{mode}")
                continue

            use_adapter = mode == "finetuned"
            if use_adapter and not args.adapter_path:
                print(f"Skipping {variant}/{mode}: --adapter_path is required")
                continue
            config = EvalConfig(
                model_path=args.model_path,
                adapter_path=args.adapter_path,
                test_data_path=args.test_data,
                output_dir=str(variant_dir),
                seed=args.seed,
                prompt_name=variant,
                limit=args.limit,
            )
            print(f"Running {variant}/{mode}...")
            results, metrics = run_evaluation(config, use_adapter=use_adapter)
            save_results(results, metrics, variant_dir, mode, config)

    rows = collect_metrics(output_dir, completed_variants, args.modes)
    write_summary(output_dir, rows, args)
    print(f"Prompt/schema robustness summary saved to: {output_dir}")


if __name__ == "__main__":
    main()
