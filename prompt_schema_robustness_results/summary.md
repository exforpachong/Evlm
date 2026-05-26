# Prompt/Schema Robustness Summary

Generated on: 2026-05-20

| Prompt | Mode | N | Strict correct | Strict acc (%) | Macro F1 | Valid JSON (%) | Delta acc vs current (pp) | Delta valid JSON vs current (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_zh | zeroshot | 211 | 187/211 | 88.63 | 0.760 | 100.00 | - | - |
| current_zh | finetuned | 211 | 195/211 | 92.42 | 0.783 | 100.00 | - | - |
| schema_en | zeroshot | 211 | 182/211 | 86.26 | 0.746 | 95.73 | -2.37 | -4.27 |
| schema_en | finetuned | 211 | 195/211 | 92.42 | 0.784 | 99.53 | +0.00 | -0.47 |

Use this table as a protocol-robustness check, not as a new model-performance claim.
