# Release README

This directory is the draft release package for the multi-disaster visual-language benchmark.

## Current Contents

| Path | Purpose |
| --- | --- |
| `DATASET_CARD.md` | Dataset scope, intended use, provenance, license status, privacy checks, and limitations. |
| `metadata_schema.json` | Machine-readable schema for public metadata and split files. |
| `release_manifest.json` | Release status, benchmark split names, known risks, and reproducibility commands. |
| `release_checklist.md` | Pre-release checklist for licensing, privacy, sensitivity, and reproducibility. |
| `metadata.json` / `metadata.jsonl` | Historical small package metadata and examples. |
| `images/` | Small example image set; not the full benchmark image release. |

## Benchmark Files Outside This Directory

Primary split files:

- `data/dataset_train_canonical_clean.json`
- `data/dataset_val_canonical_clean.json`
- `data/dataset_test_canonical_clean.json`

Robustness split files:

- `data/dataset_train_classification_full.json`
- `data/dataset_val_classification_full.json`
- `data/dataset_test_classification_full.json`
- `data/test_fallback_audit.csv`

Primary image directory:

- `sample_images/`

## Release Policy

The benchmark can be documented and reproduced locally now. The GitHub release should include code, metadata, split files, benchmark scripts, evaluation outputs, and documentation by default. Public raw-image redistribution should happen only after source provenance, license status, privacy, and sensitive-content checks are complete; web-search discovery and academic use do not by themselves grant redistribution rights. If full raw-image redistribution is not possible, release metadata, splits, scripts, and acquisition/reconstruction instructions instead, and include only cleared example or benchmark images in the public image package.

## Baseline Additions

The release now includes a reproducible image-only non-VLM calibration baseline:

```powershell
python scripts\evaluate_classical_image_baseline.py
```

This baseline trains a class-balanced ridge classifier on simple image features from the canonical clean training split and evaluates on the canonical clean test split. It is not a structured JSON generator; JSON validity is not applicable.
