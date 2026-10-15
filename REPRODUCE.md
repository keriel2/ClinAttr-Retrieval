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
  --domains <domain> --languages en,fr,es --num-pairs 300 \
  --generator-model qwen --validator-model llama
```
The validator flag matters: the script default is BioMistral, which was not used for the
reported pairs. The two deterministic filters (edit complexity, answer leakage) are applied
by the triage script that produced the screened files (likely `auto_triage_pulmonology2.py`; confirm the name and add it to this repository).

## 2. Train the gate (Orthopedic development pairs only)
```bash
python clinattr_multilingual_retrieval.py train-gate \
  --pairs-csv <orthopedic_working.csv> --train-split train \
  --norm global --feature-mode full --output-checkpoint models/gate_full.pt
python clinattr_multilingual_retrieval.py train-gate \
  --pairs-csv <orthopedic_working.csv> --train-split train \
  --norm global --feature-mode fusion_only --output-checkpoint models/gate_fusion_only.pt
```

## 3. Evaluate both directions
```bash
python clinattr_multilingual_retrieval.py evaluate-pairs \
  --pairs-csv <pairs.csv> --test-split test --gate-checkpoint models/gate_full.pt
```
`evaluate-pairs` always writes the same two output files; copy them after each run.

## 4. Exact intervals
The paper reports exact (Clopper-Pearson) intervals and an exact binomial test against .25.
The raw output of `evaluate-pairs` still contains bootstrap intervals, so recompute:
```bash
python scripts/exact_ci.py --k 5 --n 29      # k pairs consistent out of n
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
