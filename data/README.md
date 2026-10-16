# Data

Release copies of the screened counterfactual pairs. Licence: CC BY-NC 4.0 (see `../DATA_LICENSE.md`); fields: `../DATA_SCHEMA.md`.

| File | Pairs | Content |
|---|---|---|
| `orthopedic_surgery_pairs.csv` | 111 | Orthopedic Surgery; `split` is `train` (82, gate development) or `test` (29) |
| `psychiatry_psychology_pairs.csv` | 113 | Psychiatry and Psychology, all used for evaluation |
| `pulmonology_pairs.csv` | 122 | Pulmonology, all used for evaluation |
| `crossgen_llama_pairs.csv` | 54 | Cross-generator check, Llama-generated pairs |
| `crossgen_qwen_pairs.csv` | 33 | Cross-generator check, Qwen-generated pairs for the same sources (matched on language, query and passage) |

The first three files hold the 346 pairs of the main collection. The two cross-generator files are for the robustness check only.

Every pair passed the model validator and two rule-based filters (`screening_status = passed`). This is machine screening, not human adjudication; `human_adjudication` is empty until a person judges the pair. Pairs that failed screening are not released.
