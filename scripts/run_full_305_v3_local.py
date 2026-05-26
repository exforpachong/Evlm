#!/usr/bin/env python3
"""
Full 305 v3 Fair Evaluation - Local Paths
=========================================
Evaluate on dataset_test_classification_full.json using v3 protocol.
"""

import json
import torch
import time
import argparse
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)

from eval_metrics import compute_classification_metrics

# Local paths
BASE_DIR = Path(r"C:\Users\34791\Desktop\数据集")
MODEL_PATH = str(BASE_DIR / "models" / "Qwen3-VL-2B-Instruct")
ADAPTER_PATH = str(BASE_DIR / "finetune_output_v2" / "final_adapter")
TEST_DATA_PATH = str(BASE_DIR / "data" / "dataset_test_classification_full.json")
IMAGES_DIR = str(BASE_DIR / "sample_images")
OUTPUT_DIR = str(BASE_DIR / "full_305_v3_results")


@dataclass
class EvalConfig:
    model_path: str = MODEL_PATH
    adapter_path: str = ADAPTER_PATH
    test_data_path: str = TEST_DATA_PATH
    images_dir: str = IMAGES_DIR
    output_dir: str = OUTPUT_DIR
    
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    max_pixels: int = 262144
    max_new_tokens: int = 512
    temperature: float = 0.0
    do_sample: bool = False
    seed: int = 42
    
    def get_torch_dtype(self):
        return torch.bfloat16 if self.torch_dtype == "bfloat16" else torch.float16


CANONICAL_LABELS = ["flood", "earthquake", "fire", "landslide", "windstorm_or_typhoon", "other"]

# SAME PROMPT as evaluate_v3.py - critical for fair comparison
PROMPT_TEMPLATE = """你是一个灾害评估专家。观察这张图片，分析以下内容并以JSON格式回答：
1. disaster_type: 识别灾害类型 (flood/earthquake/fire/landslide/windstorm_or_typhoon/other)
2. damage_count: 损毁建筑数量/受灾人数
3. object_relations: 描述图片中主要对象的关系
4. report: 综合灾害信息，给出一份50字以内的灾情报告"""


@dataclass
class SampleResult:
    id: str
    ground_truth: str
    label_source: str  # "canonical" or "filename_fallback"
    prediction: Optional[str]
    raw_output: str
    valid_json: bool
    inference_time: float
    error: Optional[str] = None


def parse_json_output(text: str) -> Optional[dict]:
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


def load_model(config: EvalConfig, use_adapter: bool = False):
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_generation_kwargs(config: EvalConfig) -> dict:
    kwargs = {"max_new_tokens": config.max_new_tokens, "do_sample": config.do_sample}
    if config.do_sample:
        kwargs["temperature"] = config.temperature
    return kwargs


def evaluate_sample(model, processor, process_vision_info, item: dict, 
                    config: EvalConfig) -> SampleResult:
    from PIL import Image
    
    item_id = item.get('id', 'unknown')
    images = item.get('images', [])
    # label_source is in classification field
    classification = item.get('classification', {})
    label_source = classification.get('label_source', 'unknown')
    
    if not images:
        return SampleResult(
            id=item_id, ground_truth="", label_source=label_source,
            prediction=None, raw_output="", valid_json=False,
            inference_time=0, error="no_image"
        )
    
    image_path = Path(config.images_dir) / Path(images[0]).name
    
    # Get ground truth from classification.disaster_type
    gt_disaster_type = classification.get('disaster_type')
    
    # Also try conversations as backup
    if not gt_disaster_type:
        for conv in item.get('conversations', []):
            if conv.get('from') == 'assistant':
                parsed = parse_json_output(conv.get('value', ''))
                if parsed:
                    gt_disaster_type = parsed.get('disaster_type')
                break
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": PROMPT_TEMPLATE}
            ]
        }
    ]
    
    try:
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
        
        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                **build_generation_kwargs(config),
            )
        inference_time = time.time() - start_time
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0]
        
        parsed = parse_json_output(output_text)
        
        return SampleResult(
            id=item_id,
            ground_truth=gt_disaster_type or "",
            label_source=label_source,
            prediction=parsed.get('disaster_type') if parsed else None,
            raw_output=output_text[:1000],
            valid_json=parsed is not None,
            inference_time=inference_time
        )
        
    except Exception as e:
        return SampleResult(
            id=item_id, ground_truth=gt_disaster_type or "",
            label_source=label_source,
            prediction=None, raw_output="", valid_json=False,
            inference_time=0, error=str(e)
        )


def main():
    parser = argparse.ArgumentParser(description="Full 305 v3 Fair Evaluation")
    parser.add_argument("--mode", choices=["zeroshot", "finetuned", "both"], default="both")
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--adapter_path", default=ADAPTER_PATH)
    parser.add_argument("--test_data", default=TEST_DATA_PATH)
    parser.add_argument("--images_dir", default=IMAGES_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--do_sample", action="store_true",
                        help="Enable stochastic decoding. Default is deterministic greedy decoding.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature; ignored unless --do_sample is set.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    config = EvalConfig(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        test_data_path=args.test_data,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        do_sample=args.do_sample,
        temperature=args.temperature,
        seed=args.seed,
    )
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    print(f"Loading test data from: {config.test_data_path}")
    with open(config.test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples")
    
    # Count by label_source (in classification field)
    canonical_count = sum(1 for item in test_data if item.get('classification', {}).get('label_source') == 'canonical')
    fallback_count = sum(1 for item in test_data if item.get('classification', {}).get('label_source') == 'filename_fallback')
    print(f"  Canonical samples: {canonical_count}")
    print(f"  Fallback samples: {fallback_count}")
    
    modes = []
    if args.mode == "both":
        modes = ["zeroshot", "finetuned"]
    else:
        modes = [args.mode]
    
    for mode in modes:
        set_reproducible_seed(config.seed)

        print(f"\n{'='*60}")
        print(f"Running {mode.upper()} evaluation...")
        print(f"{'='*60}")
        
        use_adapter = (mode == "finetuned")
        model, processor, process_vision_info = load_model(config, use_adapter)
        
        results = []
        for i, item in enumerate(test_data):
            result = evaluate_sample(model, processor, process_vision_info, item, config)
            results.append(result)
            
            if (i + 1) % 30 == 0:
                correct = sum(1 for r in results if r.ground_truth == r.prediction)
                valid = sum(1 for r in results if r.valid_json)
                print(f"  Progress: {i+1}/{len(test_data)}, Acc: {correct/(i+1):.1%}, Valid JSON: {valid/(i+1):.1%}")
        
        # Free memory
        del model
        torch.cuda.empty_cache()
        
        # Compute metrics
        total = len(results)
        valid_json = sum(1 for r in results if r.valid_json)
        
        # Canonical vs Fallback breakdown
        canonical_results = [r for r in results if r.label_source == 'canonical']
        fallback_results = [r for r in results if r.label_source == 'filename_fallback']

        overall_metrics = compute_classification_metrics(results, CANONICAL_LABELS)
        canonical_metrics = compute_classification_metrics(canonical_results, CANONICAL_LABELS)
        fallback_metrics = compute_classification_metrics(fallback_results, CANONICAL_LABELS)
        latencies = [r.inference_time for r in results if r.inference_time > 0]
        
        metrics = {
            "model": mode,
            "total": total,
            "strict_correct": overall_metrics["strict_correct"],
            "accuracy": overall_metrics["accuracy"],
            "strict_accuracy": overall_metrics["strict_accuracy"],
            "valid_json_rate": valid_json / total if total else 0,
            "valid_json_count": valid_json,
            "canonical_prediction_count": overall_metrics["canonical_prediction_count"],
            "canonical_prediction_rate": overall_metrics["canonical_prediction_rate"],
            "noncanonical_prediction_count": overall_metrics["noncanonical_prediction_count"],
            "missing_prediction_count": overall_metrics["missing_prediction_count"],
            "valid_subset_accuracy": overall_metrics["valid_subset_accuracy"],
            "valid_subset_count": overall_metrics["valid_subset_count"],
            "canonical_samples": len(canonical_results),
            "canonical_accuracy": canonical_metrics["accuracy"],
            "canonical_strict_correct": canonical_metrics["strict_correct"],
            "canonical_macro_f1": canonical_metrics["macro_f1"],
            "canonical_weighted_f1": canonical_metrics["weighted_f1"],
            "canonical_valid_subset_accuracy": canonical_metrics["valid_subset_accuracy"],
            "canonical_valid_subset_count": canonical_metrics["valid_subset_count"],
            "fallback_samples": len(fallback_results),
            "fallback_accuracy": fallback_metrics["accuracy"],
            "fallback_strict_correct": fallback_metrics["strict_correct"],
            "fallback_macro_f1": fallback_metrics["macro_f1"],
            "fallback_weighted_f1": fallback_metrics["weighted_f1"],
            "fallback_valid_subset_accuracy": fallback_metrics["valid_subset_accuracy"],
            "fallback_valid_subset_count": fallback_metrics["valid_subset_count"],
            "macro_f1": overall_metrics["macro_f1"],
            "weighted_f1": overall_metrics["weighted_f1"],
            "avg_inference_time": float(np.mean(latencies)) if latencies else 0.0,
            "eval_config": asdict(config),
        }
        
        # Save results
        results_path = output_dir / f"{mode}_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
        
        metrics_path = output_dir / f"{mode}_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        config_path = output_dir / f"{mode}_eval_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, ensure_ascii=False, indent=2)
        
        # Print summary
        print(f"\n{mode.upper()} Results:")
        print(f"  Total: {total}")
        print(f"  Valid JSON: {valid_json} ({metrics['valid_json_rate']:.1%})")
        print(f"  Strict Accuracy: {metrics['strict_correct']}/{total} ({metrics['accuracy']:.1%})")
        print(f"  Canonical Strict Accuracy: {metrics['canonical_strict_correct']}/{len(canonical_results)} ({metrics['canonical_accuracy']:.1%})")
        print(f"  Fallback Strict Accuracy: {metrics['fallback_strict_correct']}/{len(fallback_results)} ({metrics['fallback_accuracy']:.1%})")
        print(f"  Avg Inference Time: {metrics['avg_inference_time']:.2f}s")
        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
