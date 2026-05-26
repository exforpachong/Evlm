#!/usr/bin/env python3
"""Evaluate a lightweight image-only classical baseline.

The baseline deliberately avoids pretrained VLMs. It extracts simple visual
features from each image and trains a linear classifier on the canonical clean
training split, then evaluates on the canonical clean test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import f1_score
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "baseline_results"
CANONICAL_LABELS = [
    "flood",
    "earthquake",
    "fire",
    "landslide",
    "windstorm_or_typhoon",
    "other",
]


def parse_json_output(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return None
    try:
        return json.loads(text[start_idx : end_idx + 1])
    except json.JSONDecodeError:
        return None


def load_samples(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        label = None
        for conv in item.get("conversations", []):
            if conv.get("from") != "assistant":
                continue
            parsed = parse_json_output(conv.get("value", ""))
            if parsed:
                label = parsed.get("disaster_type")
            break
        if label not in CANONICAL_LABELS:
            continue
        images = item.get("images") or []
        if not images:
            continue
        samples.append(
            {
                "id": item.get("id", ""),
                "image": images[0],
                "label": label,
            }
        )
    return samples


def _safe_open_rgb(image_path: Path) -> Image.Image:
    with Image.open(image_path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def _channel_hist(arr: np.ndarray, bins: int = 16) -> np.ndarray:
    features = []
    for channel in range(arr.shape[2]):
        hist, _ = np.histogram(arr[:, :, channel], bins=bins, range=(0, 1), density=False)
        hist = hist.astype(np.float32)
        denom = hist.sum()
        features.append(hist / denom if denom else hist)
    return np.concatenate(features)


def _gradient_hist(gray: np.ndarray, cells: int = 4, bins: int = 9) -> np.ndarray:
    grad_y, grad_x = np.gradient(gray)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    orientation = (np.arctan2(grad_y, grad_x) + np.pi) / (2 * np.pi)
    h, w = gray.shape
    cell_h = h // cells
    cell_w = w // cells
    features = []
    for row in range(cells):
        for col in range(cells):
            y0 = row * cell_h
            x0 = col * cell_w
            y1 = h if row == cells - 1 else (row + 1) * cell_h
            x1 = w if col == cells - 1 else (col + 1) * cell_w
            hist, _ = np.histogram(
                orientation[y0:y1, x0:x1],
                bins=bins,
                range=(0, 1),
                weights=magnitude[y0:y1, x0:x1],
            )
            hist = hist.astype(np.float32)
            denom = hist.sum()
            features.append(hist / denom if denom else hist)
    return np.concatenate(features)


def extract_features(image_path: Path) -> np.ndarray:
    img = _safe_open_rgb(image_path)
    width, height = img.size

    small = img.resize((32, 32), Image.Resampling.BILINEAR)
    small_arr = np.asarray(small, dtype=np.float32) / 255.0
    pixel_features = small_arr.reshape(-1)

    medium = img.resize((64, 64), Image.Resampling.BILINEAR)
    medium_arr = np.asarray(medium, dtype=np.float32) / 255.0
    rgb_hist = _channel_hist(medium_arr, bins=16)
    hsv_hist = _channel_hist(np.asarray(medium.convert("HSV"), dtype=np.float32) / 255.0, bins=16)
    gray = np.asarray(medium.convert("L"), dtype=np.float32) / 255.0
    grad_hist = _gradient_hist(gray, cells=4, bins=9)

    aspect = np.array(
        [
            width / max(height, 1),
            height / max(width, 1),
            np.log1p(width * height) / 20.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([pixel_features, rgb_hist, hsv_hist, grad_hist, aspect]).astype(np.float32)


def build_matrix(samples: list[dict[str, str]]) -> tuple[np.ndarray, list[str], list[str]]:
    features = []
    labels = []
    ids = []
    for sample in samples:
        image_path = ROOT / sample["image"]
        features.append(extract_features(image_path))
        labels.append(sample["label"])
        ids.append(sample["id"])
    return np.vstack(features), labels, ids


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [row["ground_truth"] for row in rows]
    y_pred = [row["prediction"] for row in rows]
    correct = sum(1 for gt, pred in zip(y_true, y_pred) if gt == pred)
    total = len(rows)
    return {
        "total": total,
        "strict_correct": correct,
        "accuracy": correct / total if total else 0.0,
        "strict_accuracy": correct / total if total else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=CANONICAL_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CANONICAL_LABELS, average="weighted", zero_division=0)),
        "valid_json_rate": None,
        "valid_subset_count": total,
        "valid_subset_accuracy": correct / total if total else 0.0,
        "valid_subset_macro_f1": float(
            f1_score(y_true, y_pred, labels=CANONICAL_LABELS, average="macro", zero_division=0)
        ),
        "valid_subset_weighted_f1": float(
            f1_score(y_true, y_pred, labels=CANONICAL_LABELS, average="weighted", zero_division=0)
        ),
        "note": "Image-only classical baseline; JSON validity is not applicable.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a classical image-only baseline.")
    parser.add_argument("--train", default=str(DATA_DIR / "dataset_train_canonical_clean.json"))
    parser.add_argument("--test", default=str(DATA_DIR / "dataset_test_canonical_clean.json"))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--alpha", type=float, default=1000.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_samples(Path(args.train))
    test_samples = load_samples(Path(args.test))
    print(f"Training samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print("Extracting train features...")
    x_train, y_train, _ = build_matrix(train_samples)
    print("Extracting test features...")
    x_test, y_test, test_ids = build_matrix(test_samples)

    model = make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=args.alpha, class_weight="balanced"),
    )
    print("Training ridge classifier...")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test).tolist()

    rows = []
    for sample, gt, pred in zip(test_samples, y_test, predictions):
        rows.append(
            {
                "id": sample["id"],
                "image": sample["image"],
                "ground_truth": gt,
                "prediction": pred,
                "correct": gt == pred,
                "baseline_type": "classical_image_only_ridge_classifier",
            }
        )

    metrics = compute_metrics(rows)
    metrics["config"] = {
        "train_file": str(Path(args.train)),
        "test_file": str(Path(args.test)),
        "classifier": "StandardScaler + RidgeClassifier(class_weight='balanced')",
        "features": "32x32 RGB pixels + RGB/HSV histograms + 4x4 gradient histograms + aspect features",
        "alpha": args.alpha,
        "seed": 42,
    }

    results_path = output_dir / "classical_image_baseline_results.json"
    metrics_path = output_dir / "classical_image_baseline_metrics.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.3f}")
    print(f"Wrote {results_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
