# Reproducing the reported results

All commands are run from the repository root. `--seed` is a global option and goes before
the subcommand. Paths come from the environment (see `scripts/patch_portable_paths.py`).

## 0. Setup
```bash
pip install -r requirements.txt
export CLINATTR_ROOT=/path/to/workdir
export CLINATTR_QWEN=/path/to/Qwen2.5-7B-Instruct
export CLINATTR_LLAMA=/path/to/Llama-3.1-8B-Instruct
python clinattr_multilingual_retrieval.py doctor
python clinattr_multilingual_retrieval.py prepare
```

## 1. Generate and screen pairs (slow; needs a GPU)
```bash
python clinattr_multilingual_retrieval.py generate-pairs \
  --domains orthopedic --languages en,fr,es --num-pairs 300 \
  --generator-model qwen --validator-model llama
```
The validator flag matters: the script default is BioMistral, which was not used for the
reported pairs. The two deterministic filters (edit complexity, answer leakage) are applied
by the triage script that produced the screened files (`auto_triage.py`, `auto_triage_pulmonology2.py` and `auto_triage_psychiatry.py` in this repository).

## 2. Train the gate (Orthopedic development pairs only)
```bash
python clinattr_multilingual_retrieval.py train-gate \
  --pairs-csv data/orthopedic_surgery_pairs.csv --train-split train \
  --norm global --feature-mode full --output-checkpoint models/gate_full.pt
python clinattr_multilingual_retrieval.py train-gate \
  --pairs-csv data/orthopedic_surgery_pairs.csv --train-split train \
  --norm global --feature-mode fusion_only --output-checkpoint models/gate_fusion_only.pt
```

## 3. Evaluate both directions
The released files carry `screening_status`; the evaluation script expects the working column
`approved`. Restore it first (released files contain passed pairs only):
```bash
python -c "import pandas as pd,sys; d=pd.read_csv(sys.argv[1]); d['approved']=1; d.to_csv(sys.argv[2],index=False)" \
  data/orthopedic_surgery_pairs.csv /tmp/orthopedic_eval.csv
```
Then pass the restored file to `--pairs-csv` below.

```bash
python clinattr_multilingual_retrieval.py evaluate-pairs \
  --pairs-csv /tmp/orthopedic_eval.csv --test-split test --gate-checkpoint models/gate_full.pt
```
`evaluate-pairs` always writes the same two output files; copy them after each run.

## 4. Exact intervals
The paper reports exact (Clopper-Pearson) intervals and an exact binomial test against .25.
The raw output of `evaluate-pairs` still contains bootstrap intervals, so recompute:
```bash
python scripts/exact_ci.py --k 5 --n 29      # example: 5 of 29 pairs consistent -> [.058, .358]
```

## 5. Counts and figures
```bash
python scripts/summarize_pairs.py data/*.csv --latex counts.tex
python paper/figures/make_figures.py
```

## Known gaps (also in the paper's Limitations)
- The early-stopping split inside the 82 development pairs is a random pair-level split.
- The standardization statistics include the 16 early-stopping pairs.
- Four Orthopedic passages occur in both development and test.
- Pairs are machine-screened, not human-adjudicated.

## Released files
`data/` holds `orthopedic_surgery_pairs.csv` (111 pairs: 82 `train`, 29 `test`),
`psychiatry_psychology_pairs.csv` (113), `pulmonology_pairs.csv` (122) and
`crossgen_llama_pairs.csv` and `crossgen_qwen_pairs.csv` (cross-generator check). Run step 3 on each file in turn
(Psychiatry and Pulmonology use the whole file for evaluation).
