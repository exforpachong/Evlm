# Pre-Release Checklist

## Required Before Public Release

- [ ] Confirm source provenance for each public raw image.
- [ ] Record source URL or source domain when available.
- [ ] Mark each image as `cleared`, `metadata_only`, `restricted`, or `unknown`.
- [ ] Remove raw images with unclear redistribution rights from the public image package.
- [ ] Do not upload web-searched raw images to GitHub unless redistribution rights and privacy/sensitive-content checks are complete.
- [ ] Review visible faces, license plates, addresses, signs, and other personal identifiers.
- [ ] Redact or exclude images with unresolved privacy risk.
- [ ] Flag severe injury, death, or distressing content if present.
- [ ] Confirm that `data/test_fallback_audit.csv` remains complete before reporting Full 305 metrics.
- [ ] Re-run `scripts/verify_experiment_results.py` after any split or label change.
- [ ] Re-run `scripts/evaluate_classical_image_baseline.py` before reporting the non-VLM image-only baseline.
- [ ] Re-run `scripts/evaluate_prompt_schema_robustness.py --reuse_current_main_results` before using robustness claims.
- [ ] Confirm that repository metadata contains no private identifiers or local paths.
- [ ] Confirm the final public license for metadata, code, split files, and cleared images.

## Benchmark Reporting Checks

- [ ] Keep canonical clean 211 as the primary benchmark.
- [ ] Treat Full 305 as robustness/supporting evidence.
- [ ] Do not describe simulated fallback labels as expert annotation.
- [ ] Do not claim a new model architecture or new algorithm.
- [ ] Do not present `object_relations` or `report` as fully validated generation metrics without human scoring.
- [ ] State that raw-image release depends on license, provenance, privacy, and sensitive-content review.
- [ ] State that the GitHub release can always include code/metadata/splits/scripts, but raw images are conditional on clearance.
