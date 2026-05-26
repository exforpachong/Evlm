#!/usr/bin/env python3
"""Evaluate canonical-data LoRA ablation adapters with the v3 protocol."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from evaluate_v3 import EvalConfig, run_evaluation


ROOT = Path(__file__).resolve().parent.parent


def parse_scales(value: str) -> list[str]:
    if value == "all":
        return ["25", "50", "100"]
    scales = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [scale for scale in scales if scale not in {"25", "50", "100"}]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported scale(s): {', '.join(invalid)}")
    return scales


def save_scale_results(results, metrics, output_dir: Path, scale: str, config: EvalConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"scale_{scale}pct_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump([asdict(result) for result in results], f, ensure_ascii=False, indent=2)

    metrics_payload = asdict(metrics)
    metrics_payload["scale"] = scale
    metrics_payload["adapter_path"] = config.adapter_path
    metrics_payload["eval_config"] = asdict(config)
    metrics_path = output_dir / f"scale_{scale}pct.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    config_path = output_dir / f"scale_{scale}pct_eval_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    print(f"Saved scale {scale}% results to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate canonical LoRA ablation adapters")
    parser.add_argument("--scale", type=parse_scales, default=["25", "50", "100"],
                        help="One of 25, 50, 100, comma-separated list, or all.")
    parser.add_argument("--model_path", default=str(ROOT / "models" / "Qwen3-VL-2B-Instruct"))
    parser.add_argument("--adapters_dir", default=str(ROOT / "ablation_adapters"))
    parser.add_argument("--test_data", default=str(ROOT / "data" / "dataset_test_canonical_clean.json"))
    parser.add_argument("--images_dir", default=str(ROOT / "sample_images"))
    parser.add_argument("--output_dir", default=str(ROOT / "ablation_eval_results_v3"))
    parser.add_argument("--do_sample", action="store_true",
                        help="Enable stochastic decoding. Default is deterministic greedy decoding.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature; ignored unless --do_sample is set.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    adapters_dir = Path(args.adapters_dir)
    for scale in args.scale:
        adapter_path = adapters_dir / f"scale_{scale}pct"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found for scale {scale}%: {adapter_path}")

        print(f"\nEvaluating canonical LoRA ablation scale {scale}%")
        config = EvalConfig(
            model_path=args.model_path,
            adapter_path=str(adapter_path),
            test_data_path=args.test_data,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            do_sample=args.do_sample,
            temperature=args.temperature,
            seed=args.seed,
        )
        results, metrics = run_evaluation(config, use_adapter=True)
        save_scale_results(results, metrics, output_dir, scale, config)


if __name__ == "__main__":
    main()
