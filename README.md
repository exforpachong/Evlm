# Multi-Disaster Visual-Language Benchmark

This repository provides data splits, metadata, evaluation code, and baseline results for a multi-disaster visual-language benchmark.

## Benchmark Summary

| Item | Value |
| --- | --- |
| Dataset type | Multi-disaster visual-language image benchmark |
| Primary task | Canonical disaster-type prediction under a structured JSON output contract |
| Canonical labels | `flood`, `earthquake`, `fire`, `landslide`, `windstorm_or_typhoon`, `other` |
| Canonical clean split | 1,714 train, 216 validation, 211 test |
| Full classification split | 2,453 train, 306 validation, 305 test |
| Main evaluation | Strict accuracy, macro F1, weighted F1, JSON validity |
| VLM baselines | Qwen3-VL-2B zero-shot, Qwen3-VL-2B LoRA, InternVL3-2B zero-shot |
| Non-VLM baseline | Classical image-only ridge classifier |

Only `disaster_type` is quantitatively validated in the current benchmark. The structured output also includes `damage_count`, `object_relations`, and `report`; these fields are provided for structured-output research but should not be treated as fully validated generation metrics without additional human scoring.

## Repository Contents

Recommended public repository layout:

| Path | Purpose |
| --- | --- |
| `README.md` | Repository homepage |
| `data/dataset_train_canonical_clean.json` | Primary training split |
| `data/dataset_val_canonical_clean.json` | Primary validation split |
| `data/dataset_test_canonical_clean.json` | Primary test split |
| `data/dataset_train_classification_full.json` | Robustness training split |
| `data/dataset_val_classification_full.json` | Robustness validation split |
| `data/dataset_test_classification_full.json` | Robustness test split |
| `data/test_fallback_audit.csv` | Human-reviewed fallback labels for the full 305-image test set |
| `data/canonical_clean_statistics.json` | Canonical split statistics |
| `data/classification_full_statistics.json` | Full split statistics |
| `release_package/DATASET_CARD.md` | Dataset scope, intended use, release policy, and limitations |
| `release_package/RELEASE_README.md` | Release package notes |
| `release_package/metadata_schema.json` | Machine-readable public metadata schema |
| `release_package/release_manifest.json` | Release status and reproducibility commands |
| `release_package/release_checklist.md` | License, privacy, sensitivity, and reproducibility checklist |
| `scripts/download_real_disasters.py` | Image acquisition workflow |
| `scripts/filter_images.py` | Image filtering workflow |
| `scripts/audit_approved_dataset.py` | Dataset audit utility |
| `scripts/build_canonical_dataset.py` | Canonical dataset construction |
| `scripts/build_canonical_clean.py` | Canonical clean split construction |
| `scripts/build_classification_full.py` | Full classification split construction |
| `scripts/data_audit.py` | Split and data-integrity audit |
| `scripts/fix_data_leakage.py` | Duplicate/leakage audit support |
| `scripts/eval_metrics.py` | Shared metric utilities |
| `scripts/evaluate_v3.py` | Main Qwen3-VL-2B evaluation |
| `scripts/evaluate_full_305_v3.py` | Full 305-image robustness evaluation |
| `scripts/evaluate_internvl3_baseline.py` | InternVL3-2B baseline evaluation |
| `scripts/evaluate_prompt_schema_robustness.py` | Prompt/schema robustness evaluation |
| `scripts/evaluate_ablation_v3.py` | Data-scale ablation evaluation |
| `scripts/evaluate_classical_image_baseline.py` | Classical image-only baseline |
| `scripts/calculate_baselines.py` | Majority and stratified-random baselines |
| `scripts/paired_significance_test.py` | Paired McNemar significance testing |
| `scripts/generate_publication_artifacts.py` | Result table generation |
| `scripts/verify_experiment_results.py` | Raw-result verification |
| `eval_results_v3/` | Qwen3-VL-2B zero-shot and LoRA raw predictions and metrics |
| `internvl3_eval_results/` | InternVL3-2B zero-shot raw predictions and metrics |
| `baseline_results/` | Majority, random, and classical image-only baseline results |
| `prompt_schema_robustness_results/` | Prompt/schema robustness results |
| `ablation_eval_results_v3_retrained_20260514/` | Retrained 50% and 100% data-scale ablation results |

The repository should contain only the data, code, metadata, and derived benchmark outputs needed to reproduce the results. Local planning notes, writing files, private credentials, local paths, full model weights, and unreviewed raw images should be kept outside the public repository.

## Data Files

Primary benchmark splits:

```text
data/dataset_train_canonical_clean.json
data/dataset_val_canonical_clean.json
data/dataset_test_canonical_clean.json
```

Robustness splits:

```text
data/dataset_train_classification_full.json
data/dataset_val_classification_full.json
data/dataset_test_classification_full.json
data/test_fallback_audit.csv
```

Dataset-level statistics:

```text
data/canonical_clean_statistics.json
data/classification_full_statistics.json
```

## Core Code

Data construction and auditing:

```text
scripts/download_real_disasters.py
scripts/filter_images.py
scripts/audit_approved_dataset.py
scripts/build_canonical_dataset.py
scripts/build_canonical_clean.py
scripts/build_classification_full.py
scripts/data_audit.py
scripts/fix_data_leakage.py
```

Model evaluation:

```text
scripts/evaluate_v3.py
scripts/evaluate_full_305_v3.py
scripts/evaluate_internvl3_baseline.py
scripts/evaluate_prompt_schema_robustness.py
scripts/evaluate_ablation_v3.py
scripts/evaluate_classical_image_baseline.py
scripts/eval_metrics.py
```

Result verification:

```text
scripts/calculate_baselines.py
scripts/paired_significance_test.py
scripts/generate_publication_artifacts.py
scripts/verify_experiment_results.py
```

## Reproducibility Commands

Regenerate result tables and verification reports:

```powershell
python scripts\generate_publication_artifacts.py
python scripts\verify_experiment_results.py
```

Run the classical image-only baseline:

```powershell
python scripts\evaluate_classical_image_baseline.py
```

Run the main Qwen3-VL-2B benchmark, assuming the base model and LoRA adapter are available locally:

```powershell
python scripts\evaluate_v3.py --mode both --adapter_path finetune_output_v2\final_adapter
```

Run the full 305-image robustness evaluation:

```powershell
python scripts\run_full_305_v3_local.py --mode both
```

Run the InternVL3-2B zero-shot baseline, assuming the model is available locally:

```powershell
python scripts\evaluate_internvl3_baseline.py --model_path models\InternVL3-2B-hf --output_dir internvl3_eval_results --local_files_only
```

Run prompt/schema robustness:

```powershell
python scripts\evaluate_prompt_schema_robustness.py --reuse_current_main_results
```

Run paired significance testing:

```powershell
python scripts\paired_significance_test.py
```

## Main Results

| Scope | Model | N | Strict accuracy | Macro F1 | Weighted F1 | Valid JSON |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Canonical clean test | Classical image-only ridge classifier | 211 | 59.72% | 0.538 | 0.602 | N/A |
| Canonical clean test | Qwen3-VL-2B zero-shot | 211 | 88.63% | 0.760 | 0.892 | 100.00% |
| Canonical clean test | Qwen3-VL-2B LoRA | 211 | 92.42% | 0.783 | 0.918 | 100.00% |
| Canonical clean test | InternVL3-2B zero-shot | 211 | 91.00% | 0.771 | 0.902 | 100.00% |
| Full 305 test | Qwen3-VL-2B zero-shot | 305 | 85.90% | 0.784 | 0.861 | 100.00% |
| Full 305 test | Qwen3-VL-2B LoRA | 305 | 87.54% | 0.748 | 0.861 | 100.00% |

Paired McNemar testing on the canonical clean 211-image test set shows that Qwen3-VL-2B LoRA improves over Qwen3-VL-2B zero-shot by 3.79 percentage points, with exact `p = 0.03857`.

## Prompt and Schema Robustness

| Prompt | Mode | N | Strict accuracy | Macro F1 | Valid JSON |
| --- | --- | ---: | ---: | ---: | ---: |
| Current Chinese structured prompt | Qwen3 zero-shot | 211 | 88.63% | 0.760 | 100.00% |
| Current Chinese structured prompt | Qwen3 LoRA | 211 | 92.42% | 0.783 | 100.00% |
| Constrained English schema prompt | Qwen3 zero-shot | 211 | 86.26% | 0.746 | 95.73% |
| Constrained English schema prompt | Qwen3 LoRA | 211 | 92.42% | 0.784 | 99.53% |

## Data-Scale Ablation

| Training scale | N | Strict accuracy | Macro F1 | Valid JSON |
| --- | ---: | ---: | ---: | ---: |
| 25% canonical training data | 211 | 91.00% | 0.808 | 100.00% |
| 50% canonical training data | 211 | 92.89% | 0.853 | 100.00% |
| 100% canonical training data | 211 | 92.89% | 0.846 | 100.00% |

## JSON Output Contract

Each model is asked to return a valid JSON object with the following fields:

```json
{
  "disaster_type": "flood",
  "damage_count": "unknown",
  "object_relations": "Flooded roadway with vehicles surrounded by standing water.",
  "report": "The image shows flood damage affecting road traffic and nearby infrastructure."
}
```

The accepted canonical labels are:

```json
[
  "flood",
  "earthquake",
  "fire",
  "landslide",
  "windstorm_or_typhoon",
  "other"
]
```

Strict evaluation treats invalid JSON, missing predictions, and non-canonical `disaster_type` values as errors.

## Raw Image Release Policy

The benchmark can be shared with metadata, split files, scripts, and evaluation outputs. Public raw-image redistribution should happen only after each image has been checked for:

- Source provenance
- Redistribution license
- Visible faces, license plates, addresses, or other identifiers
- Sensitive disaster content
- Source-domain restrictions

If a raw image cannot be redistributed, keep the metadata record, split membership, label, and reconstruction or acquisition instructions where legally appropriate.

## Known Limitations

- The canonical clean test set contains 211 images, which limits statistical power.
- The `other` class has only five canonical clean test samples and should be treated as an out-of-distribution diagnostic class.
- `object_relations` and `report` are structured fields, but they have not yet received independent human scoring.
- Split-integrity auditing found no test-involved image overlap, but small train-validation duplicates should be removed in future releases.
- Raw-image redistribution depends on per-image license, privacy, and sensitive-content review.

