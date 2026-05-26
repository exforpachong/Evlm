#!/usr/bin/env python3
"""Evaluate InternVL3-2B as an additional open VLM baseline.

The script mirrors the canonical-clean evaluation protocol used by
``evaluate_v3.py`` but keeps model loading separate because InternVL3 uses the
generic Hugging Face image-text-to-text API instead of the Qwen3-VL class.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval_metrics import CANONICAL_LABELS, compute_classification_metrics
from evaluate_v3 import PROMPT_TEMPLATE, extract_damage_count, parse_json_output


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class InternVLConfig:
    model_path: str = "OpenGVLab/InternVL3-2B-hf"
    test_data_path: str = "data/dataset_test_canonical_clean.json"
    images_dir: str = "sample_images"
    output_dir: str = "internvl3_eval_results"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0
    seed: int = 42
    limit: int | None = None
    local_files_only: bool = False
    model_card: str = "https://huggingface.co/OpenGVLab/InternVL3-2B-hf"
    license: str = "qwen"
    note: str = "OpenGVLab InternVL3-2B Hugging Face image-text-to-text baseline."

    def get_torch_dtype(self) -> torch.dtype:
        if self.torch_dtype == "float16":
            return torch.float16
        if self.torch_dtype == "float32":
            return torch.float32
        return torch.bfloat16


@dataclass
class SampleResult:
    id: str
    ground_truth: str
    prediction: str | None
    raw_output: str
    valid_json: bool
    parsed_json: dict[str, Any] | None
    inference_time: float
    peak_memory_gb: float
    error: str | None = None


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_generation_kwargs(config: InternVLConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
    }
    if config.do_sample:
        kwargs["temperature"] = config.temperature
    return kwargs


def load_model(config: InternVLConfig):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = config.get_torch_dtype()
    print(f"Loading InternVL3 baseline from: {config.model_path}")
    processor = AutoProcessor.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        local_files_only=config.local_files_only,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_path,
        torch_dtype=dtype,
        device_map=config.device_map,
        trust_remote_code=True,
        local_files_only=config.local_files_only,
    )
    model.eval()
    return model, processor


def load_test_data(config: InternVLConfig) -> list[dict[str, Any]]:
    path = ROOT / config.test_data_path
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if config.limit is not None:
        rows = rows[: config.limit]
    return rows


def ground_truth_from_item(item: dict[str, Any]) -> tuple[str, int | None]:
    gt_disaster_type = ""
    gt_damage_count = None
    for conv in item.get("conversations", []):
        if conv.get("from") == "assistant":
            parsed = parse_json_output(conv.get("value", ""))
            if parsed:
                gt_disaster_type = parsed.get("disaster_type") or ""
                gt_damage_count = extract_damage_count(parsed.get("damage_count"))
            break
    return gt_disaster_type, gt_damage_count


def image_reference_candidates(image_path: Path) -> list[dict[str, str]]:
    resolved = image_path.resolve()
    return [
        {"type": "image", "url": resolved.as_uri()},
        {"type": "image", "path": str(resolved)},
        {"type": "image", "image": str(resolved)},
    ]


def prepare_inputs(processor, image_path: Path, config: InternVLConfig):
    last_error: Exception | None = None
    for image_ref in image_reference_candidates(image_path):
        messages = [
            {
                "role": "user",
                "content": [
                    image_ref,
                    {"type": "text", "text": PROMPT_TEMPLATE},
                ],
            }
        ]
        try:
            return processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as exc:  # pragma: no cover - depends on processor variant
            last_error = exc
    if last_error:
        raise last_error
    raise RuntimeError("No image reference candidates were available.")


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def evaluate_sample(model, processor, item: dict[str, Any], config: InternVLConfig) -> SampleResult:
    item_id = item.get("id", "unknown")
    images = item.get("images") or []
    gt_disaster_type, _ = ground_truth_from_item(item)
    if not images:
        return SampleResult(
            id=item_id,
            ground_truth=gt_disaster_type,
            prediction=None,
            raw_output="",
            valid_json=False,
            parsed_json=None,
            inference_time=0.0,
            peak_memory_gb=0.0,
            error="no_image",
        )

    image_path = ROOT / config.images_dir / Path(images[0]).name
    if not image_path.exists():
        return SampleResult(
            id=item_id,
            ground_truth=gt_disaster_type,
            prediction=None,
            raw_output="",
            valid_json=False,
            parsed_json=None,
            inference_time=0.0,
            peak_memory_gb=0.0,
            error=f"missing_image: {image_path}",
        )

    try:
        inputs = prepare_inputs(processor, image_path, config)
        inputs = inputs.to(model_device(model), dtype=config.get_torch_dtype())

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                **build_generation_kwargs(config),
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time = time.time() - start_time

        input_length = inputs["input_ids"].shape[-1]
        output_text = processor.decode(
            generated_ids[0][input_length:],
            skip_special_tokens=True,
        )
        parsed = parse_json_output(output_text)
        peak_memory = (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available()
            else 0.0
        )
        return SampleResult(
            id=item_id,
            ground_truth=gt_disaster_type,
            prediction=parsed.get("disaster_type") if parsed else None,
            raw_output=output_text[:1000],
            valid_json=parsed is not None,
            parsed_json=parsed,
            inference_time=inference_time,
            peak_memory_gb=peak_memory,
        )
    except Exception as exc:
        return SampleResult(
            id=item_id,
            ground_truth=gt_disaster_type,
            prediction=None,
            raw_output="",
            valid_json=False,
            parsed_json=None,
            inference_time=0.0,
            peak_memory_gb=0.0,
            error=str(exc),
        )


def summarize_latency_memory(rows: list[SampleResult]) -> dict[str, float]:
    latencies = [row.inference_time for row in rows if row.inference_time > 0]
    memories = [row.peak_memory_gb for row in rows if row.peak_memory_gb > 0]
    if not latencies:
        return {
            "avg_latency": 0.0,
            "median_latency": 0.0,
            "p95_latency": 0.0,
            "peak_gpu_memory_gb": max(memories) if memories else 0.0,
        }
    return {
        "avg_latency": float(np.mean(latencies)),
        "median_latency": float(np.median(latencies)),
        "p95_latency": float(np.percentile(latencies, 95)),
        "peak_gpu_memory_gb": max(memories) if memories else 0.0,
    }


def aggregate_metrics(results: list[SampleResult], config: InternVLConfig) -> dict[str, Any]:
    classification = compute_classification_metrics(results, CANONICAL_LABELS)
    latency_memory = summarize_latency_memory(results)
    return {
        "model": "InternVL3-2B zero-shot",
        "total_samples": len(results),
        "strict_correct_count": classification["strict_correct"],
        "disaster_type_accuracy": classification["accuracy"],
        **classification,
        **latency_memory,
        "field_completion_rate": field_completion_rate(results),
        "eval_config": asdict(config),
    }


def field_completion_rate(results: list[SampleResult]) -> dict[str, float]:
    fields = ["disaster_type", "damage_count", "object_relations", "report"]
    valid = [row for row in results if row.valid_json and row.parsed_json]
    rates = {}
    for field in fields:
        rates[field] = (
            sum(1 for row in valid if field in (row.parsed_json or {})) / len(valid)
            if valid
            else 0.0
        )
    return rates


def save_results(results: list[SampleResult], metrics: dict[str, Any], config: InternVLConfig) -> None:
    output_dir = ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "zeroshot_results.json"
    metrics_path = output_dir / "zeroshot_metrics.json"
    config_path = output_dir / "zeroshot_eval_config.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump([asdict(row) for row in results], f, ensure_ascii=False, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)
    print(f"Saved results to {results_path.relative_to(ROOT)}")
    print(f"Saved metrics to {metrics_path.relative_to(ROOT)}")


def parse_args() -> InternVLConfig:
    parser = argparse.ArgumentParser(description="Evaluate InternVL3-2B open VLM baseline.")
    parser.add_argument("--model_path", default="OpenGVLab/InternVL3-2B-hf")
    parser.add_argument("--test_data", default="data/dataset_test_canonical_clean.json")
    parser.add_argument("--images_dir", default="sample_images")
    parser.add_argument("--output_dir", default="internvl3_eval_results")
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test sample limit.")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()
    return InternVLConfig(
        model_path=args.model_path,
        test_data_path=args.test_data,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        seed=args.seed,
        limit=args.limit,
        local_files_only=args.local_files_only,
    )


def main() -> None:
    config = parse_args()
    set_reproducible_seed(config.seed)
    rows = load_test_data(config)
    print(f"Loaded {len(rows)} canonical-clean samples")
    model, processor = load_model(config)

    results: list[SampleResult] = []
    for idx, item in enumerate(rows, start=1):
        result = evaluate_sample(model, processor, item, config)
        results.append(result)
        if idx % 20 == 0 or idx == len(rows):
            metrics = compute_classification_metrics(results, CANONICAL_LABELS)
            print(
                f"  {idx}/{len(rows)} acc={metrics['accuracy']:.2%} "
                f"valid_json={metrics['valid_json_rate']:.2%}"
            )

    metrics = aggregate_metrics(results, config)
    save_results(results, metrics, config)
    print(
        "InternVL3-2B zero-shot: "
        f"acc={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}, "
        f"valid_json={metrics['valid_json_rate']:.2%}, "
        f"avg_latency={metrics['avg_latency']:.2f}s, "
        f"peak_gpu={metrics['peak_gpu_memory_gb']:.2f}GB"
    )


if __name__ == "__main__":
    main()
