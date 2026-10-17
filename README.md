# ClinAttr-CF

A multilingual clinical counterfactual test collection for evaluating attribute-sensitive retrieval, with the construction pipeline, evaluation code, and diagnostic baselines.

ClinAttr-CF is built from [CUREv1](https://huggingface.co/datasets/clinia/CUREv1). Each pair keeps a real clinical passage unchanged and creates an edited copy that differs by one grounded span, for example a changed disease stage, dose, or negation, together with a query rewritten to fit the edited passage. A retriever is credited only when it prefers the original passage for the original query and reverses that preference for the edited query.

> Status: the pairs are **machine-screened**, not human-adjudicated. See `DATA_SCHEMA.md` (`screening_status`, `human_adjudication`) and the paper's Limitations section.

## What is in this repository

| Path | Content |
|---|---|
| `clinattr_multilingual_retrieval.py` | Pipeline and evaluation (generation, validation, triage, gate training, evaluation) |
| `data/` | Released pair files (see `data/README.md`) |
| `scripts/summarize_pairs.py` | Pair counts by domain, language, and attribute family (CSV and LaTeX) |
| `scripts/exact_ci.py` | Exact (Clopper-Pearson) interval and exact binomial test against .25 |
| `scripts/patch_portable_paths.py` | Replaces hard-coded server paths with environment variables |
| `scripts/prepublish_check.sh` | Scans the working tree for secrets, absolute paths, and large files |
| `tools/annotation/` | Human audit sheet builder and agreement script, with a practice workbook |
| `paper/figures/` | Figure script and the figures used in the paper |
| `DATA_SCHEMA.md` | Field definitions |
| `MODELS.md` | Model identifiers and inference settings |
| `REPRODUCE.md` | Commands that regenerate the reported results |

## Quick start

```bash
pip install -r requirements.txt
export CLINATTR_ROOT=/path/to/workdir          # data, caches, outputs
export CLINATTR_QWEN=/path/to/Qwen2.5-7B-Instruct
export CLINATTR_LLAMA=/path/to/Llama-3.1-8B-Instruct
python clinattr_multilingual_retrieval.py doctor
```

Full commands, including the evaluation that produces the paper tables, are in `REPRODUCE.md`.

## Intended use

Testing whether a retriever or reranker changes its ranking when one clinically decisive attribute changes, and training or probing models on such changes. The edited passages are synthetic. Some may be clinically implausible or contradict other parts of the text, and none should be used as medical information. The collection is not intended for clinical decision making.

## Licences

- Code: MIT (`LICENSE`).
- Pairs and derived text: CC BY-NC 4.0 (`DATA_LICENSE.md`), because the passages and queries derive from CUREv1, whose dataset card states CC BY-NC 4.0.

## Citation

See `CITATION.cff`. 

## Contact and corrections

Open an issue on this repository for corrections. Maintainer: Henry Mukalazi Serugunda, School of Computer Science, Xiangtan University, Xiangtan, Hunan, China, and Department of Information Technology, Makerere University, Kampala, Uganda. Email: henry.serugunda@mak.ac.ug
