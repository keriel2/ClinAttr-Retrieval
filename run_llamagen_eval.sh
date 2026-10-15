#!/usr/bin/env bash
# Does the main finding hold on pairs from a different generator?
# Scores (a) the Llama-generated Pulmonology pairs and (b) the Qwen-generated pairs for the SAME
# 100 source examples, with the same systems and the same global normalization as the main results.
# The learned gate here is the one trained on the 82 Orthopedic training pairs (not retrained).
# Run:  bash run_llamagen_eval.sh 2>&1 | tee llamagen_eval.log
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
cd /home/share/henry/CURE

D=/home/share/henry/dataset/clinattr_project
C=$D/counterfactuals
R=$D/results
PYB=/root/anaconda3/envs/lin/bin/python
PY="$PYB clinattr_multilingual_retrieval.py"
DENSE=cuda:6
LATE=cuda:2

echo "=== 1. Machine triage of the Llama-generated pairs (copy, original untouched) ==="
$PYB triage_copy.py "$C/counterfactual_candidates_llamagen.csv" "$C/llamagen_triaged.csv"

echo "=== 2. Matched Qwen-generated pairs (same 100 source examples, existing approvals) ==="
$PYB - <<'PYEOF'
import pandas as pd
C = "/home/share/henry/dataset/clinattr_project/counterfactuals/"
llama = pd.read_csv(C + "counterfactual_candidates_llamagen.csv")
qwen = pd.read_csv(C + "pulmonology_batch_backup.csv")
sub = qwen[qwen["pair_id"].isin(set(llama["pair_id"]))].copy()
sub.to_csv(C + "qwengen_matched100.csv", index=False)
print("matched Qwen rows:", len(sub), "| approved:", int((sub["approved"].astype(str).isin(["1","True","true"])).sum()))
PYEOF

run_eval () {
  local pairs=$1 tag=$2
  echo "=== Evaluating $tag ==="
  $PY --seed 42 evaluate-pairs \
    --pairs-csv "$pairs" \
    --gate-checkpoint "$R/gate_full_global.pt" \
    --dense-device "$DENSE" --late-device "$LATE" --gate-device cpu
  cp "$R/counterfactual_pair_scores.csv" "$R/scores_full_global_${tag}.csv"
  cp "$R/counterfactual_capr_summary.csv" "$R/summary_full_global_${tag}.csv"
}

run_eval "$C/qwengen_matched100.csv" pulm_qwengen100
run_eval "$C/llamagen_triaged.csv"   pulm_llamagen100

echo "=== 3. Side by side ==="
$PYB - <<'PYEOF'
import pandas as pd
R = "/home/share/henry/dataset/clinattr_project/results/"
rows = []
for tag, name in [("pulm_qwengen100", "Qwen-generated"), ("pulm_llamagen100", "Llama-generated")]:
    df = pd.read_csv(R + f"summary_full_global_{tag}.csv")
    df = df[df["scope"] == "overall"]
    for _, r in df.iterrows():
        rows.append({"pairs": name, "method": r["method"], "n": int(r["n"]),
                     "A_o": round(r["capr"], 3), "A_c": round(r["capr_counterfactual_direction"], 3),
                     "balanced": round(r["capr_balanced"], 3),
                     "PC": round(r["pair_consistency"], 3),
                     "PC_low": round(r["pair_consistency_ci95_low"], 3),
                     "PC_high": round(r["pair_consistency_ci95_high"], 3)})
out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(R + "generator_robustness_pc.csv", index=False)
print("saved", R + "generator_robustness_pc.csv")
PYEOF
