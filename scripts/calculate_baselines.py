#!/usr/bin/env python3
"""
Calculate Majority and Random baselines for the canonical clean benchmark.
"""

import json
import random
import argparse
from collections import Counter
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "baseline_results"

CANONICAL_LABELS = ["flood", "earthquake", "fire", "landslide", "windstorm_or_typhoon", "other"]


def parse_json_output(text: str):
    if not text:
        return None
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return None
    try:
        return json.loads(text[start_idx:end_idx + 1])
    except json.JSONDecodeError:
        return None


def load_labels(data_path):
    """Load disaster_type labels from dataset."""
    labels = []
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        for conv in item.get('conversations', []):
            if conv.get('from') == 'assistant':
                parsed = parse_json_output(conv.get('value', ''))
                label = parsed.get('disaster_type') if parsed else None
                if label in CANONICAL_LABELS:
                    labels.append(label)
                break
    return labels, data

def main():
    parser = argparse.ArgumentParser(description="Calculate simple baselines")
    parser.add_argument("--train", default=str(DATA_DIR / "dataset_train_canonical_clean.json"))
    parser.add_argument("--test", default=str(DATA_DIR / "dataset_test_canonical_clean.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load training labels to find majority class
    print("Loading training data...")
    train_labels, _ = load_labels(Path(args.train))
    print(f"Training samples: {len(train_labels)}")
    
    # Count distribution
    train_dist = Counter(train_labels)
    print("\nTraining distribution:")
    for label, count in train_dist.most_common():
        print(f"  {label}: {count} ({count/len(train_labels)*100:.1f}%)")
    
    # Find majority class
    majority_class = train_dist.most_common(1)[0][0]
    print(f"\nMajority class: {majority_class}")
    
    # Load test data
    print("\nLoading test data...")
    test_labels, test_data = load_labels(Path(args.test))
    print(f"Test samples: {len(test_labels)}")
    
    # Test distribution
    test_dist = Counter(test_labels)
    print("\nTest distribution:")
    for label, count in test_dist.most_common():
        print(f"  {label}: {count} ({count/len(test_labels)*100:.1f}%)")
    
    # Majority baseline
    majority_correct = sum(1 for label in test_labels if label == majority_class)
    majority_accuracy = majority_correct / len(test_labels)
    
    print(f"\n{'='*50}")
    print("MAJORITY BASELINE")
    print(f"{'='*50}")
    print(f"Predicted class: {majority_class}")
    print(f"Correct: {majority_correct}/{len(test_labels)}")
    print(f"Accuracy: {majority_accuracy:.4f} ({majority_accuracy*100:.2f}%)")
    
    # Random baseline (stratified - predict according to training distribution)
    random.seed(args.seed)
    classes = list(train_dist.keys())
    weights = [train_dist[c] / len(train_labels) for c in classes]
    
    random_correct = 0
    for true_label in test_labels:
        pred_label = random.choices(classes, weights=weights)[0]
        if pred_label == true_label:
            random_correct += 1
    
    random_accuracy = random_correct / len(test_labels)
    
    print(f"\n{'='*50}")
    print("RANDOM BASELINE (Stratified)")
    print(f"{'='*50}")
    print(f"Classes: {classes}")
    print(f"Weights: {[f'{w:.3f}' for w in weights]}")
    print(f"Correct: {random_correct}/{len(test_labels)}")
    print(f"Accuracy: {random_accuracy:.4f} ({random_accuracy*100:.2f}%)")
    
    # Save results
    results = {
        "majority_baseline": {
            "predicted_class": majority_class,
            "correct": majority_correct,
            "total": len(test_labels),
            "accuracy": majority_accuracy
        },
        "random_baseline": {
            "classes": classes,
            "weights": weights,
            "correct": random_correct,
            "total": len(test_labels),
            "accuracy": random_accuracy
        },
        "training_distribution": dict(train_dist),
        "test_distribution": dict(test_dist),
        "config": {
            "train_file": str(Path(args.train)),
            "test_file": str(Path(args.test)),
            "seed": args.seed,
        },
    }
    
    with open(OUTPUT_DIR / "baseline_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {OUTPUT_DIR / 'baseline_results.json'}")
    
    # Summary comparison
    print(f"\n{'='*50}")
    print("BASELINE COMPARISON")
    print(f"{'='*50}")
    print(f"{'Method':<30} {'Accuracy':>15}")
    print("-" * 45)
    print(f"{'Majority Baseline':<30} {majority_accuracy:>15.2%}")
    print(f"{'Random Baseline (Stratified)':<30} {random_accuracy:>15.2%}")

if __name__ == "__main__":
    main()
