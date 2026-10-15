#!/usr/bin/env python3
"""
rename_screening_fields.py   (run on COPIES of the pair files made for the release)

The working files use  approved=1  for pairs that passed the model validator and the two
deterministic filters. For the public release that wording suggests human approval. This
writes a release copy with
    screening_status      passed | failed        (machine outcome)
    screening_method      llm_validator+deterministic_filters
    human_adjudication    empty until a person judges the pair
and drops  approved  and  reviewer.

Do NOT run this on the files the evaluation script reads: it expects an  approved  column.

Usage: python3 rename_screening_fields.py in.csv out.csv
"""
import sys
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)
ok = df["approved"].astype(str).str.lower().isin(["1", "true", "yes"])
df["screening_status"] = ok.map({True: "passed", False: "failed"})
df["screening_method"] = "llm_validator+deterministic_filters"
df["human_adjudication"] = ""
df = df.drop(columns=[c for c in ("approved", "reviewer") if c in df.columns])
df.to_csv(dst, index=False)
print(f"{dst}: {int(ok.sum())} passed, {int((~ok).sum())} failed; approved/reviewer dropped")
