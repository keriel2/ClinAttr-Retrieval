#!/usr/bin/env bash
# Second-generator check: Llama-3.1-8B-Instruct proposes the edits, Qwen2.5-7B-Instruct validates.
# Uses the same shuffled source list as the original Pulmonology run (seed 42), first 100 examples,
# so pair_id values match and the two runs can be compared pair by pair.
# Output goes to counterfactual_candidates_llamagen.csv; existing files are not touched.
set -e
cd /home/share/henry/CURE
export HF_ENDPOINT=https://hf-mirror.com
PY=/root/anaconda3/envs/lin/bin/python

# 1. patch once (safe to re-run only if it has not been applied)
grep -q "out-suffix" clinattr_multilingual_retrieval.py || $PY add_out_suffix_patch.py clinattr_multilingual_retrieval.py

# 2. generate and validate
$PY clinattr_multilingual_retrieval.py generate-pairs \
  --domains pulmonology --languages en,fr,es --num-pairs 100 \
  --generator-model llama --validator-model qwen \
  --generator-device cuda:6 --validator-device cuda:2 \
  --out-suffix _llamagen

# 3. compare with the original Qwen-generated Pulmonology candidates
$PY compare_generators.py \
  /home/share/henry/dataset/clinattr_project/counterfactuals/pulmonology_batch_backup.csv \
  /home/share/henry/dataset/clinattr_project/counterfactuals/counterfactual_candidates_llamagen.csv \
  --out /home/share/henry/dataset/clinattr_project/results/generator_comparison.csv
