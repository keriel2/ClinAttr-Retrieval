#!/usr/bin/env bash
# run_global_norm_eval.sh
#
# Same experiment as run_symmetric_eval.sh, but the gates are trained with
# --norm global, so dense and late scores keep their magnitudes instead of
# collapsing to +1/-1. Requires add_global_norm_patch.py to have been applied.
#
# Reuses the feature cache written by the previous training run (same 82
# Orthopedic training pairs), so no passages are re-scored during training.
# Results are saved under the tags full_global and fusion_only_global, so the
# earlier pairwise-normalization results stay in place as the "before" rows.
#
# Run with:
#   nohup bash run_global_norm_eval.sh > global_norm_run.log 2>&1 &
#   tail -f global_norm_run.log

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

echo "=== Step 1: train gates with global normalization ==="

$PY --seed 42 train-gate \
  --pairs-csv "$ORTHO" --train-split train \
  --feature-mode full --norm global \
  --output-checkpoint "$R/gate_full_global.pt" \
  --dense-device "$DENSE" --late-device "$LATE" --gate-device "$DENSE" \
  --resume

$PY --seed 42 train-gate \
  --pairs-csv "$ORTHO" --train-split train \
  --feature-mode fusion_only --norm global \
  --output-checkpoint "$R/gate_fusion_only_global.pt" \
  --dense-device "$DENSE" --late-device "$LATE" --gate-device "$DENSE" \
  --resume

echo "=== Learned weights ==="
cat "$R/gate_full_global.weights.json"; echo
cat "$R/gate_fusion_only_global.weights.json"; echo

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

for GATE in full_global fusion_only_global; do
  run_eval "$GATE" "$ORTHO" ortho_test --test-split test
  run_eval "$GATE" "$PULM"  pulm
  run_eval "$GATE" "$PSYCH" psych
done

echo "=== Step 3: overview and near-tie report ==="

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

print("\nNear-ties in gate decisions (score gap below 1e-5, decided by rounding noise):")
for f in sorted(glob.glob(R + "scores_full*_*.csv")):
    df = pd.read_csv(f)
    tag = os.path.basename(f)[len("scores_"):-4]
    t1 = int((df["attr_gate_margin"].abs() < 1e-5).sum())
    t2 = int((df["attr_gate_c_margin"].abs() < 1e-5).sum())
    print(f"  {tag}: original direction {t1}/{len(df)}, counterfactual direction {t2}/{len(df)}")
print("\nSaved:", R + "symmetric_capr_overview.csv")
PYEOF
