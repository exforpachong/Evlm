# Dataset Card: Multi-Disaster Visual-Language Benchmark

Version: 1.0 draft release package  
Last updated: 2026-05-20

## Intended Use

This dataset package supports research on visual-language disaster assessment for emergency management. The primary benchmark task is canonical disaster-type classification from an image under a structured JSON output contract. The structured output schema also includes `damage_count`, `object_relations`, and `report` fields, but only `disaster_type` is quantitatively validated in the current release.

Recommended uses:

- Evaluate lightweight vision-language models on multi-disaster image assessment.
- Test structured JSON validity and canonical label compliance.
- Reproduce the canonical clean benchmark and robustness analyses.
- Study prompt/schema robustness under fixed decoding and fixed splits.
- Compare VLM results with a lightweight image-only classical baseline.

Not recommended uses:

- Automated emergency response without human review.
- Claims about fully validated generated reports or object-relation reasoning before human scoring is added.
- High-stakes damage estimation from `damage_count` without additional expert validation.

## Dataset Scope

The benchmark uses a six-class canonical taxonomy:

- `flood`
- `earthquake`
- `fire`
- `landslide`
- `windstorm_or_typhoon`
- `other`

Current split sizes:

| Dataset version | Train | Validation | Test | Role |
| --- | ---: | ---: | ---: | --- |
| Canonical clean | 1,714 | 216 | 211 | Primary benchmark |
| Classification full | 2,453 | 306 | 305 | Robustness/supporting evidence |

The canonical clean test set is the primary benchmark. The full 305-image test set includes 94 filename-fallback test samples whose labels have been manually reviewed; it should be treated as supporting robustness evidence.

## Data Sources and Provenance

Images were collected through multi-keyword disaster image search and local filtering. Search terms cover flood, earthquake, fire, landslide, typhoon, hurricane, cyclone, tornado, storm surge, and related disaster aftermath scenes. The project scripts record the collection and filtering process, but individual image redistribution rights must be checked before public raw-image release.

Relevant local scripts:

- `scripts/download_real_disasters.py`
- `scripts/filter_images.py`
- `scripts/audit_approved_dataset.py`
- `scripts/build_canonical_clean.py`
- `scripts/build_classification_full.py`

Release rule:

- Publicly release raw images only after source provenance and license status are cleared.
- If some images cannot be redistributed, release metadata, split files, image identifiers, preprocessing scripts, benchmark scripts, and reproducible acquisition instructions instead.

## Annotation Process

The annotation pipeline uses GLM-4.6V automatic visual-language annotation followed by manual review and canonical schema normalization. Labels that could not be parsed into canonical JSON were excluded from the canonical clean benchmark. The full classification split retains filename-fallback labels for robustness analysis; the 94 fallback test labels were manually reviewed before reporting Full 305 metrics.

Current known label policy:

- Canonical clean split: parsed canonical labels from reviewed structured annotations.
- Full test fallback subset: human-reviewed fallback labels stored in `data/test_fallback_audit.csv`.
- `data/test_fallback_audit_simulated.csv` is an AI-assisted simulation reference only and must not be described as expert annotation.

## JSON Schema

Each benchmark sample follows a conversation-style visual-language format with an image path and a target assistant JSON object.

Required target fields:

- `disaster_type`: one of the six canonical labels.
- `damage_count`: integer, numeric phrase, or unknown when visible count is unclear.
- `object_relations`: concise description of visible object relations and damage cues.
- `report`: concise disaster assessment intended for human review.

The current quantitative benchmark validates `disaster_type`, JSON validity, and canonical label compliance. Human scoring is still needed before `object_relations` and `report` are used as validated generation metrics.

## Privacy and Sensitive Content

Disaster imagery may include damaged homes, vehicles, infrastructure, rescue scenes, and potentially identifiable people or private property. Before public raw-image release:

- Review images for visible faces, license plates, addresses, and other personal identifiers.
- Remove or blur sensitive personal information where needed.
- Flag severe injury, death, or distressing content if present.
- Exclude images with unclear redistribution rights.
- Document any redaction or exclusion decisions in the release manifest.

## License Status

Dataset metadata, benchmark splits, scripts, and derived evaluation outputs can be prepared for open release under a project-selected open license. Raw image licensing is currently unresolved and must be handled per image or per source before redistribution. Images found through web search are not automatically redistributable on GitHub, even when used for academic research.

Suggested release structure:

- Code and metadata: permissive research license selected by the project owner.
- Raw images on GitHub: only images with confirmed redistribution rights and completed privacy/sensitive-content checks.
- Restricted images: metadata-only entries with acquisition instructions or source references.
- Unclear images: do not upload raw files; release image identifiers, source-domain/source-URL fields when available, split membership, and reconstruction scripts/instructions.

## Reproducibility

Primary artifact regeneration commands:

```powershell
python scripts\evaluate_classical_image_baseline.py
python scripts\generate_publication_artifacts.py
python scripts\verify_experiment_results.py
```

Canonical clean Qwen3 evaluation:

```powershell
python scripts\evaluate_v3.py --mode both --adapter_path finetune_output_v2\final_adapter
```

Prompt/schema robustness:

```powershell
python scripts\evaluate_prompt_schema_robustness.py --reuse_current_main_results
```

InternVL3-2B baseline:

```powershell
python scripts\evaluate_internvl3_baseline.py --model_path models\InternVL3-2B-hf --output_dir internvl3_eval_results --local_files_only
```

## Known Limitations

- The canonical clean test set contains 211 images; larger test sets would improve statistical power.
- The `other` class has only five canonical clean test samples and should be treated as an out-of-distribution diagnostic category.
- Train-validation image MD5 overlaps are known in the current split; no test-involved overlap was found.
- Raw-image release is conditional on license, provenance, privacy, and sensitive-content review.
- Generated `object_relations` and `report` fields are not yet human-scored.
- The classical image-only baseline is a calibration reference, not a deployment model and not a structured JSON generator.

## Maintainer Information

Maintainer information is omitted from the anonymous review release.
