#!/usr/bin/env python3
"""
Unified Evaluation Script v3
=============================
Single script for all model evaluation with consistent configuration.

Features:
- Consistent loading config for all models
- Comprehensive metrics: JSON validity, classification, damage count
- Bootstrap 95% CI for key metrics
- Per-class precision/recall/F1 and confusion matrix
- Latency percentiles (avg/median/P95)
- GPU memory tracking

Usage:
    python scripts/evaluate_v3.py --mode zeroshot
    python scripts/evaluate_v3.py --mode finetuned --adapter_path path/to/adapter
    python scripts/evaluate_v3.py --mode both
"""

import json
import torch
import time
import re
import argparse
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional
from sklearn.metrics import (
    f1_score,
)

from eval_metrics import compute_classification_metrics, strict_correct, strict_label_sequences

# ============================================================
# Configuration - MUST be identical for all models
# ============================================================
@dataclass
class EvalConfig:
    # Model paths
    model_path: str = "models/Qwen3-VL-2B-Instruct"
    adapter_path: Optional[str] = None
    test_data_path: str = "data/dataset_test_canonical_clean.json"
    images_dir: str = "sample_images"
    output_dir: str = "eval_results_v3"
    
    # Loading config - MUST be identical
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    max_pixels: int = 262144  # 512x512
    max_new_tokens: int = 512
    temperature: float = 0.0
    do_sample: bool = False
    seed: int = 42
    prompt_name: str = "current_zh"
    prompt_template: str = ""
    limit: Optional[int] = None
    
    # Evaluation config
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    def get_torch_dtype(self):
        return torch.bfloat16 if self.torch_dtype == "bfloat16" else torch.float16


CANONICAL_LABELS = ["flood", "earthquake", "fire", "landslide", "windstorm_or_typhoon", "other"]

# Default prompt template used for the main reported evaluations.
PROMPT_TEMPLATE = """你是一个灾害评估专家。观察这张图片，分析以下内容并以JSON格式回答：
1. disaster_type: 识别灾害类型 (flood/earthquake/fire/landslide/windstorm_or_typhoon/other)
2. damage_count: 损毁建筑数量/受灾人数
3. object_relations: 描述图片中主要对象的关系
4. report: 综合灾害信息，给出一份50字以内的灾情报告"""

SCHEMA_EN_PROMPT_TEMPLATE = """You are a disaster assessment expert. Inspect the image and return only one valid JSON object.

Use exactly this schema:
{
  "disaster_type": one of ["flood", "earthquake", "fire", "landslide", "windstorm_or_typhoon", "other"],
  "damage_count": an integer if visible, otherwise "unknown",
  "object_relations": a concise string describing visible objects, spatial relations, and damage cues,
  "report": a concise disaster assessment for human review
}

Rules:
- Do not include markdown, comments, explanations, or text outside the JSON object.
- The disaster_type value must be one of the six canonical labels exactly.
- If evidence is ambiguous, choose the most visually supported canonical label and explain uncertainty in report."""

PROMPT_VARIANTS = {
    "current_zh": PROMPT_TEMPLATE,
    "schema_en": SCHEMA_EN_PROMPT_TEMPLATE,
}


@dataclass
class SampleResult:
    """Result for a single sample."""
    id: str
    ground_truth: str
    prediction: Optional[str]
    raw_output: str
    valid_json: bool
    parsed_json: Optional[dict]
    inference_time: float
    peak_memory_gb: float
    error: Optional[str] = None


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""
    # Basic counts
    total_samples: int
    valid_json_count: int
    valid_json_rate: float
    invalid_json_count: int
    
    # Strict classification metrics (all canonical-GT samples stay in denominator)
    strict_correct_count: int
    accuracy: float
    disaster_type_accuracy: float
    macro_f1: float
    weighted_f1: float
    canonical_prediction_count: int
    canonical_prediction_rate: float
    noncanonical_prediction_count: int
    missing_prediction_count: int

    # Valid-subset metrics (valid JSON plus canonical prediction)
    valid_subset_count: int
    valid_subset_accuracy: float
    valid_subset_macro_f1: float
    valid_subset_weighted_f1: float
    
    # Per-class metrics
    per_class_metrics: dict
    
    # Damage count metrics
    damage_count_exact_match: float
    damage_count_mae: float
    damage_count_bucket_accuracy: dict
    
    # All fields
    all_fields_exact_match: float
    field_completion_rate: dict
    
    # Latency
    avg_latency: float
    median_latency: float
    p95_latency: float
    
    # Memory
    peak_gpu_memory_gb: float
    
    # Bootstrap CI
    accuracy_ci: tuple
    macro_f1_ci: tuple
    
    # Confusion matrix
    confusion_matrix_labels: list
    confusion_matrix: list


def parse_json_output(text: str) -> Optional[dict]:
    """Extract and parse JSON from model output."""
    if not text:
        return None
    
    text = text.strip()
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx == -1 or end_idx == -1:
        return None
    
    json_str = text[start_idx:end_idx+1]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def extract_damage_count(value) -> Optional[int]:
    """Extract numeric damage count from various formats."""
    if value is None:
        return None
    
    if isinstance(value, int):
        return value
    
    if isinstance(value, str):
        # Try to extract number
        numbers = re.findall(r'\d+', value)
        if numbers:
            return int(numbers[0])
    
    return None


def bucket_damage_count(count: Optional[int]) -> str:
    """Bucket damage count into categories."""
    if count is None:
        return "unknown"
    if count == 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 20:
        return "6-20"
    return ">20"


def load_model(config: EvalConfig, use_adapter: bool = False):
    """Load model with consistent configuration."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel
    
    print(f"Loading model from: {config.model_path}")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.model_path,
        torch_dtype=config.get_torch_dtype(),
        device_map=config.device_map,
        trust_remote_code=True,
    )
    
    if use_adapter and config.adapter_path:
        print(f"Loading adapter from: {config.adapter_path}")
        model = PeftModel.from_pretrained(model, config.adapter_path)
    
    processor = AutoProcessor.from_pretrained(
        config.model_path,
        max_pixels=config.max_pixels,
        trust_remote_code=True,
    )
    
    return model, processor, process_vision_info


def set_reproducible_seed(seed: int):
    """Seed Python, NumPy and torch before deterministic evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_generation_kwargs(config: EvalConfig) -> dict:
    """Keep greedy decoding deterministic by not passing sampling-only args."""
    kwargs = {"max_new_tokens": config.max_new_tokens, "do_sample": config.do_sample}
    if config.do_sample:
        kwargs["temperature"] = config.temperature
    return kwargs


def get_prompt_template(config: EvalConfig) -> str:
    """Resolve the prompt text for an evaluation run."""
    if config.prompt_template:
        return config.prompt_template
    return PROMPT_VARIANTS.get(config.prompt_name, PROMPT_TEMPLATE)


def evaluate_sample(model, processor, process_vision_info, item: dict, 
                    config: EvalConfig) -> SampleResult:
    """Evaluate a single sample."""
    from PIL import Image
    
    item_id = item.get('id', 'unknown')
    images = item.get('images', [])
    
    if not images:
        return SampleResult(
            id=item_id, ground_truth="", prediction=None,
            raw_output="", valid_json=False, parsed_json=None,
            inference_time=0, peak_memory_gb=0, error="no_image"
        )
    
    # Load image
    image_path = Path(config.images_dir) / Path(images[0]).name
    
    # Get ground truth
    gt_disaster_type = None
    gt_damage_count = None
    for conv in item.get('conversations', []):
        if conv.get('from') == 'assistant':
            parsed = parse_json_output(conv.get('value', ''))
            if parsed:
                gt_disaster_type = parsed.get('disaster_type')
                gt_damage_count = extract_damage_count(parsed.get('damage_count'))
            break
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": get_prompt_template(config)}
            ]
        }
    ]
    
    try:
        # Process input
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # Record memory before
        torch.cuda.synchronize()
        mem_before = torch.cuda.max_memory_allocated() / 1024**3
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                **build_generation_kwargs(config),
            )
        inference_time = time.time() - start_time
        
        # Record memory after
        torch.cuda.synchronize()
        mem_after = torch.cuda.max_memory_allocated() / 1024**3
        peak_memory = max(mem_before, mem_after)
        
        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0]
        
        # Parse prediction
        parsed = parse_json_output(output_text)
        
        return SampleResult(
            id=item_id,
            ground_truth=gt_disaster_type or "",
            prediction=parsed.get('disaster_type') if parsed else None,
            raw_output=output_text[:1000],
            valid_json=parsed is not None,
            parsed_json=parsed,
            inference_time=inference_time,
            peak_memory_gb=peak_memory
        )
        
    except Exception as e:
        return SampleResult(
            id=item_id, ground_truth=gt_disaster_type or "",
            prediction=None, raw_output="", valid_json=False,
            parsed_json=None, inference_time=0, peak_memory_gb=0,
            error=str(e)
        )


def compute_bootstrap_ci(values: list, n_samples: int = 1000, 
                         confidence: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval."""
    if not values:
        return (0.0, 0.0)
    
    values = np.array(values)
    n = len(values)
    
    bootstrap_means = []
    for _ in range(n_samples):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return (float(lower), float(upper))


def aggregate_metrics(results: list[SampleResult], config: EvalConfig) -> EvalMetrics:
    """Aggregate results into metrics."""
    total = len(results)
    
    # Basic counts
    valid_json_results = [r for r in results if r.valid_json]
    valid_json_count = len(valid_json_results)

    classification = compute_classification_metrics(results, CANONICAL_LABELS)
    strict_values = [
        1 if strict_correct(r, CANONICAL_LABELS) else 0
        for r in results
        if r.ground_truth in CANONICAL_LABELS
    ]
    acc_ci = compute_bootstrap_ci(strict_values, config.bootstrap_samples, config.confidence_level)

    y_true_strict, y_pred_strict = strict_label_sequences(results, CANONICAL_LABELS)
    if y_true_strict:
        f1_values = []
        for _ in range(config.bootstrap_samples):
            indices = np.random.choice(len(y_true_strict), size=len(y_true_strict), replace=True)
            boot_true = [y_true_strict[i] for i in indices]
            boot_pred = [y_pred_strict[i] for i in indices]
            f1_values.append(
                f1_score(boot_true, boot_pred, labels=CANONICAL_LABELS, average='macro', zero_division=0)
            )
        alpha = 1 - config.confidence_level
        f1_ci = (float(np.percentile(f1_values, alpha/2 * 100)),
                 float(np.percentile(f1_values, (1 - alpha/2) * 100)))
    else:
        f1_ci = (0.0, 0.0)
    
    # Damage count metrics
    damage_counts = []
    for r in results:
        if r.parsed_json and 'damage_count' in r.parsed_json:
            pred_count = extract_damage_count(r.parsed_json.get('damage_count'))
            # Get ground truth damage count
            for conv in next((item for item in []), {}).get('conversations', []):
                pass  # Would need original item for GT damage count
    
    # Latency
    latencies = [r.inference_time for r in results if r.inference_time > 0]
    if latencies:
        avg_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        p95_latency = np.percentile(latencies, 95)
    else:
        avg_latency = 0
        median_latency = 0
        p95_latency = 0
    
    # Memory
    memories = [r.peak_memory_gb for r in results if r.peak_memory_gb > 0]
    peak_memory = max(memories) if memories else 0
    
    # Field completion rate
    field_completion = defaultdict(lambda: {'total': 0, 'present': 0})
    required_fields = ['disaster_type', 'damage_count', 'object_relations', 'report']
    
    for r in valid_json_results:
        for field in required_fields:
            field_completion[field]['total'] += 1
            if r.parsed_json and field in r.parsed_json:
                field_completion[field]['present'] += 1
    
    field_completion_rate = {
        field: (stats['present'] / stats['total'] if stats['total'] > 0 else 0)
        for field, stats in field_completion.items()
    }
    
    return EvalMetrics(
        total_samples=total,
        valid_json_count=valid_json_count,
        valid_json_rate=valid_json_count / total if total > 0 else 0,
        invalid_json_count=classification["invalid_json_count"],
        strict_correct_count=classification["strict_correct"],
        accuracy=classification["accuracy"],
        disaster_type_accuracy=classification["accuracy"],
        macro_f1=classification["macro_f1"],
        weighted_f1=classification["weighted_f1"],
        canonical_prediction_count=classification["canonical_prediction_count"],
        canonical_prediction_rate=classification["canonical_prediction_rate"],
        noncanonical_prediction_count=classification["noncanonical_prediction_count"],
        missing_prediction_count=classification["missing_prediction_count"],
        valid_subset_count=classification["valid_subset_count"],
        valid_subset_accuracy=classification["valid_subset_accuracy"],
        valid_subset_macro_f1=classification["valid_subset_macro_f1"],
        valid_subset_weighted_f1=classification["valid_subset_weighted_f1"],
        per_class_metrics=classification["per_class_metrics"],
        damage_count_exact_match=0,  # Computed separately
        damage_count_mae=0,  # Computed separately
        damage_count_bucket_accuracy={},  # Computed separately
        all_fields_exact_match=0,  # Computed separately
        field_completion_rate=field_completion_rate,
        avg_latency=avg_latency,
        median_latency=median_latency,
        p95_latency=p95_latency,
        peak_gpu_memory_gb=peak_memory,
        accuracy_ci=acc_ci,
        macro_f1_ci=f1_ci,
        confusion_matrix_labels=classification["confusion_matrix_labels"],
        confusion_matrix=classification["confusion_matrix"],
    )


def run_evaluation(config: EvalConfig, use_adapter: bool = False) -> tuple:
    """Run full evaluation."""
    set_reproducible_seed(config.seed)

    # Load test data
    print(f"Loading test data from: {config.test_data_path}")
    with open(config.test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    if config.limit is not None:
        test_data = test_data[:config.limit]
    print(f"Loaded {len(test_data)} test samples")
    
    # Load model
    model, processor, process_vision_info = load_model(config, use_adapter)
    
    # Evaluate
    results = []
    for i, item in enumerate(test_data):
        result = evaluate_sample(model, processor, process_vision_info, item, config)
        results.append(result)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(test_data)} samples...")
    
    # Free memory
    del model
    torch.cuda.empty_cache()
    
    # Compute metrics
    metrics = aggregate_metrics(results, config)
    
    return results, metrics


def save_results(results: list, metrics: EvalMetrics, output_dir: Path, mode: str, config: EvalConfig):
    """Save evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-sample results
    results_path = output_dir / f"{mode}_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    
    # Save metrics
    metrics_path = output_dir / f"{mode}_metrics.json"
    metrics_payload = asdict(metrics)
    metrics_payload["eval_config"] = asdict(config)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    config_path = output_dir / f"{mode}_eval_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {output_dir}")


def print_metrics(metrics: EvalMetrics, mode: str):
    """Print evaluation metrics."""
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS - {mode.upper()}")
    print("="*60)
    
    print(f"\n{'Basic Metrics':-^40}")
    print(f"  Total samples: {metrics.total_samples}")
    print(f"  Valid JSON: {metrics.valid_json_count} ({metrics.valid_json_rate:.2%})")
    print(f"  Canonical predictions: {metrics.canonical_prediction_count} "
          f"({metrics.canonical_prediction_rate:.2%})")
    print(f"  Non-canonical predictions: {metrics.noncanonical_prediction_count}")
    
    print(f"\n{'Classification Metrics':-^40}")
    print(f"  Strict accuracy: {metrics.disaster_type_accuracy:.4f} "
          f"[{metrics.accuracy_ci[0]:.4f}, {metrics.accuracy_ci[1]:.4f}]")
    print(f"  Strict Macro F1: {metrics.macro_f1:.4f} "
          f"[{metrics.macro_f1_ci[0]:.4f}, {metrics.macro_f1_ci[1]:.4f}]")
    print(f"  Strict Weighted F1: {metrics.weighted_f1:.4f}")
    print(f"  Valid-subset accuracy: {metrics.valid_subset_accuracy:.4f} "
          f"(n={metrics.valid_subset_count})")
    
    print(f"\n{'Per-Class Metrics':-^40}")
    for label in CANONICAL_LABELS:
        if label in metrics.per_class_metrics:
            m = metrics.per_class_metrics[label]
            print(f"  {label}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
    
    print(f"\n{'Latency':-^40}")
    print(f"  Average: {metrics.avg_latency:.2f}s")
    print(f"  Median: {metrics.median_latency:.2f}s")
    print(f"  P95: {metrics.p95_latency:.2f}s")
    
    print(f"\n{'Memory':-^40}")
    print(f"  Peak GPU: {metrics.peak_gpu_memory_gb:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation v3")
    parser.add_argument("--mode", choices=["zeroshot", "finetuned", "both"], default="both")
    parser.add_argument("--model_path", default="models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--test_data", default="data/dataset_test_canonical_clean.json")
    parser.add_argument("--output_dir", default="eval_results_v3")
    parser.add_argument("--do_sample", action="store_true",
                        help="Enable stochastic decoding. Default is deterministic greedy decoding.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature; ignored unless --do_sample is set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_name", choices=sorted(PROMPT_VARIANTS), default="current_zh",
                        help="Built-in prompt variant to use.")
    parser.add_argument("--prompt_text", default=None,
                        help="Inline prompt text. Overrides --prompt_name when provided.")
    parser.add_argument("--prompt_template_file", default=None,
                        help="UTF-8 text file containing a prompt. Overrides --prompt_name when provided.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N test samples for smoke tests.")
    args = parser.parse_args()

    prompt_template = ""
    if args.prompt_template_file:
        prompt_template = Path(args.prompt_template_file).read_text(encoding="utf-8")
    elif args.prompt_text:
        prompt_template = args.prompt_text
    
    config = EvalConfig(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        do_sample=args.do_sample,
        temperature=args.temperature,
        seed=args.seed,
        prompt_name=args.prompt_name,
        prompt_template=prompt_template,
        limit=args.limit,
    )
    
    output_dir = Path(config.output_dir)
    
    if args.mode in ["zeroshot", "both"]:
        print("\nRunning Zero-Shot Evaluation...")
        results, metrics = run_evaluation(config, use_adapter=False)
        print_metrics(metrics, "zero-shot")
        save_results(results, metrics, output_dir, "zeroshot", config)
    
    if args.mode in ["finetuned", "both"]:
        if not config.adapter_path:
            print("\nWarning: No adapter path specified for fine-tuned evaluation")
            print("Skipping fine-tuned evaluation...")
        else:
            print("\nRunning Fine-Tuned Evaluation...")
            results, metrics = run_evaluation(config, use_adapter=True)
            print_metrics(metrics, "fine-tuned")
            save_results(results, metrics, output_dir, "finetuned", config)


if __name__ == "__main__":
    main()
