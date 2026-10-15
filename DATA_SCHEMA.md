# Data schema

One row per pair. The names below are the ones the pipeline writes. Run
`python scripts/summarize_pairs.py data/<file>.csv --schema` to print the exact header of a
file and compare it with this list; update this document if they differ.

| Field | Meaning |
|---|---|
| `pair_id` | Stable identifier of the pair |
| `domain` | CUREv1 medical domain |
| `language` | Language of the query (en, fr, es) |
| `attribute_type` | Attribute family: negation, dosage, frequency, subtype, stage, temporal, severity |
| `original_query` | CUREv1 query |
| `counterfactual_query` | Query rewritten to fit the edited passage (does not state the new value) |
| `positive_passage` | Unmodified CUREv1 passage (the name comes from the training objective) |
| `negative_passage` | Passage with the single span replaced |
| `source_span` | Verbatim span copied from the passage |
| `replacement_span` | Span that replaces it, changing one attribute value |
| `generator_model` | Model that proposed the edit |
| `split` | `train` or `test` (Orthopedic only; the paper calls the train partition "development"; other domains are evaluation only; confirm the values in the file) |
| `screening_status` | `passed` or `failed`: outcome of the model validator and the two deterministic filters |
| `screening_method` | `llm_validator+deterministic_filters` |
| `human_adjudication` | Empty unless a person judged the pair (see `tools/annotation/`) |

Notes:
- `screening_status = passed` means machine-screened, not human-approved.
- The working files used by the evaluation script call the screening outcome `approved`;
  the release files do not.
- Validator output fields written by the pipeline are kept as extra columns if present.
