"""Applies the same automated triage as auto_triage_pulmonology2.py, but to a COPY.

Usage: python3 triage_copy.py INPUT_CSV OUTPUT_CSV
Rows accepted by the validator are approved (approved=1, reviewer='auto') unless a filter
flags them: (1) the edit changes two or more separate word chunks, (2) the counterfactual
query contains the counterfactual or original attribute value. The input file is not changed.
These pairs are machine-screened, not human-reviewed.
"""
import difflib, sys
import pandas as pd

src_path, out_path = sys.argv[1], sys.argv[2]

def chunks(a, b):
    sm = difflib.SequenceMatcher(None, a.split(), b.split())
    return sum(1 for op in sm.get_opcodes() if op[0] != "equal")

def leaks(query, value):
    value = str(value).strip().lower()
    return len(value) >= 3 and value in str(query).lower()

df = pd.read_csv(src_path)
df["approved"] = 0
df["review_notes"] = df.get("review_notes", "")
acc = df["validator_decision"] == "accept_candidate"
n_flag = 0
for i, r in df[acc].iterrows():
    why = []
    c = chunks(str(r.get("source_span", "")), str(r.get("replacement_span", "")))
    if c >= 2:
        why.append(f"multi_concept_edit(chunks={c})")
    if leaks(r.get("counterfactual_query", ""), r.get("counterfactual_value", "")):
        why.append("leaks_counterfactual_value")
    if leaks(r.get("counterfactual_query", ""), r.get("original_value", "")):
        why.append("leaks_original_value")
    df.at[i, "reviewer"] = "auto"
    if why:
        n_flag += 1
        df.at[i, "review_notes"] = "; ".join(why)
    else:
        df.at[i, "approved"] = 1
print(f"accepted by validator: {int(acc.sum())}; flagged: {n_flag}; passes triage: {int((df.approved==1).sum())}")
df.to_csv(out_path, index=False)
print("saved", out_path)
