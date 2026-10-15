# Data

Place the release copies of the pair files here:

| File | Content |
|---|---|
| `orthopedic_surgery_pairs.csv` | Orthopedic Surgery, with the `split` column (train or test) |
| `psychiatry_psychology_pairs.csv` | Psychiatry and Psychology, all evaluation |
| `pulmonology_pairs.csv` | Pulmonology, all evaluation |
| `crossgen_pulmonology_pairs.csv` | Cross-generator check (Qwen-generated and Llama-generated subsets) |

Create each file from the working copy with
`python tools/annotation/rename_screening_fields.py working.csv data/<name>.csv`.
That replaces `approved` with `screening_status`, `screening_method`, and an empty
`human_adjudication` column. Never run it on the files the evaluation script reads.

Licence: CC BY-NC 4.0, see `../DATA_LICENSE.md`. Fields: `../DATA_SCHEMA.md`.
