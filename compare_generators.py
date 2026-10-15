"""Compares two candidate files built from the same source pairs by different generators.

Usage:
  python3 compare_generators.py QWEN_CSV LLAMA_CSV [--out generator_comparison.csv]

Rows are matched on pair_id (a hash of language, query id and passage id, so it does not
depend on the generator). Only matched pairs are compared.
Note: the two files also differ in validator, so acceptance differences mix generator and
validator effects. Proposal rate, attribute agreement and span agreement do not.
"""
import argparse, re
import pandas as pd

def norm(s):
    return re.sub(r"\s+", " ", str(s or "").strip().lower())

def toks(s):
    return set(re.findall(r"\w+", norm(s)))

def jaccard(a, b):
    a, b = toks(a), toks(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0

ap = argparse.ArgumentParser()
ap.add_argument("qwen_csv"); ap.add_argument("llama_csv")
ap.add_argument("--out", default="generator_comparison.csv")
a = ap.parse_args()

q = pd.read_csv(a.qwen_csv).drop_duplicates("pair_id").set_index("pair_id")
l = pd.read_csv(a.llama_csv).drop_duplicates("pair_id").set_index("pair_id")
ids = q.index.intersection(l.index)
q, l = q.loc[ids], l.loc[ids]
n = len(ids)
print(f"matched source pairs: {n} (file 1: {len(q)} rows total kept, file 2: {len(l)})")
if n == 0:
    raise SystemExit("no overlapping pair_id values; check domains, languages, seed and --num-pairs")

def produced(df):
    return df["source_span"].fillna("").astype(str).str.strip() != ""

def accepted(df):
    return df["validator_decision"].astype(str) == "accept_candidate"

pq, pl = produced(q), produced(l)
both = pq & pl
same_attr = (q["attribute_type"].astype(str) == l["attribute_type"].astype(str))[both]
span_j = pd.Series([jaccard(q.loc[i, "source_span"], l.loc[i, "source_span"]) for i in ids[both]], index=ids[both])
same_span = pd.Series([norm(q.loc[i, "source_span"]) == norm(l.loc[i, "source_span"]) for i in ids[both]], index=ids[both])

rows = [
    ("matched source pairs", n),
    ("proposal produced, file 1", int(pq.sum())),
    ("proposal produced, file 2", int(pl.sum())),
    ("both produced a proposal", int(both.sum())),
    ("same attribute family (of both produced)", f"{same_attr.mean():.3f}" if both.any() else "NA"),
    ("identical source span (of both produced)", f"{same_span.mean():.3f}" if both.any() else "NA"),
    ("source span token overlap >= 0.5 (of both produced)", f"{(span_j >= 0.5).mean():.3f}" if both.any() else "NA"),
    ("accepted by validator, file 1", int(accepted(q).sum())),
    ("accepted by validator, file 2", int(accepted(l).sum())),
    ("accepted in both", int((accepted(q) & accepted(l)).sum())),
]
out = pd.DataFrame(rows, columns=["measure", "value"])
print(out.to_string(index=False))
out.to_csv(a.out, index=False)
print("saved", a.out)
