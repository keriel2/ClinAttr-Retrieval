#!/usr/bin/env bash
# run_symmetric_eval.sh
#
# 1. Retrains the full gate and the fusion-only gate on the 82 Orthopedic
#    training pairs only (split created earlier in orthopedic_surgery_split.csv).
# 2. Evaluates both gates with the symmetric metrics on:
#      - the 29 held-out Orthopedic test pairs
#      - Pulmonology (never used in training)
#      - Psychiatry & Psychology (never used in training)
# 3. Copies each run's output to its own file, because evaluate-pairs always
#    writes to the same two filenames and would otherwise overwrite itself.
# 4. Prints one overview table at the end.
#
# Uses GPUs 2 and 3 because GPUs 0 and 1 were busy when this was written.
#
# Run with:
#   nohup bash run_symmetric_eval.sh > symmetric_run.log 2>&1 &
#   tail -f symmetric_run.log

set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com

cd /home/share/henry/CURE

D=/home/share/henry/dataset/clinattr_project
C=$D/counterfactuals
R=$D/results
PY="/root/anaconda3/envs/lin/bin/python clinattr_multilingual_retrieval.py"
DENSE=cuda:4
LATE=cuda:6

ORTHO=$C/orthopedic_surgery_split.csv
PULM=$C/pulmonology_batch_backup.csv
PSYCH=$C/psychiatry_triaged_final.csv

mkdir -p "$R"

echo "=== Step 1: retrain gates on Orthopedic training split only ==="

$PY --seed 42 train-gate \
  --pairs-csv "$ORTHO" --train-split train \
  --feature-mode full \
  --output-checkpoint "$R/gate_full.pt" \
  --dense-device "$DENSE" --late-device "$LATE" --gate-device "$DENSE"

# --resume reuses the feature cache from the run above (same pairs, same features).
$PY --seed 42 train-gate \
  --pairs-csv "$ORTHO" --train-split train \
  --feature-mode fusion_only \
  --output-checkpoint "$R/gate_fusion_only.pt" \
  --dense-device "$DENSE" --late-device "$LATE" --gate-device "$DENSE" \
  --resume

run_eval () {
  local gate=$1 pairs=$2 tag=$3
  shift 3
  echo "=== Evaluating gate=$gate on $tag ==="
  $PY --seed 42 evaluate-pairs \
    --pairs-csv "$pairs" \
    --gate-checkpoint "$R/gate_${gate}.pt" \
    --dense-device "$DENSE" --late-device "$LATE" --gate-device cpu \
    "$@"
  cp "$R/counterfactual_pair_scores.csv" "$R/scores_${gate}_${tag}.csv"
  cp "$R/counterfactual_capr_summary.csv" "$R/summary_${gate}_${tag}.csv"
}

echo "=== Step 2: symmetric evaluation ==="

for GATE in full fusion_only; do
  run_eval "$GATE" "$ORTHO" ortho_test --test-split test
  run_eval "$GATE" "$PULM"  pulm
  run_eval "$GATE" "$PSYCH" psych
done

echo "=== Step 3: overview ==="

/root/anaconda3/envs/lin/bin/python - <<'PYEOF'
import glob, os
import pandas as pd

R = "/home/share/henry/dataset/clinattr_project/results/"
rows = []
for f in sorted(glob.glob(R + "summary_*_*.csv")):
    df = pd.read_csv(f)
    df = df[df["scope"] == "overall"]
    tag = os.path.basename(f)[len("summary_"):-4]
    for _, r in df.iterrows():
        rows.append({
            "run": tag,
            "method": r["method"],
            "n": int(r["n"]),
            "original_query_direction": round(r["capr"], 3),
            "counterfactual_query_direction": round(r["capr_counterfactual_direction"], 3),
            "balanced": round(r["capr_balanced"], 3),
            "pair_consistency": round(r["pair_consistency"], 3),
        })
out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(R + "symmetric_capr_overview.csv", index=False)
print("\nSaved:", R + "symmetric_capr_overview.csv")
PYEOF
