import pandas as pd
import difflib

PATH = "/home/share/henry/dataset/clinattr_project/counterfactuals/counterfactual_candidates.csv"

def count_diff_chunks(a: str, b: str) -> int:
    sm = difflib.SequenceMatcher(None, a.split(), b.split())
    return sum(1 for op in sm.get_opcodes() if op[0] != "equal")

def leaks_answer(cf_query: str, value: str) -> bool:
    value = str(value).strip().lower()
    if not value or len(value) < 3:
        return False
    return value in str(cf_query).lower()

df = pd.read_csv(PATH)
accepted_mask = df["validator_decision"] == "accept_candidate"

flags_by_pair = {}
for idx, row in df[accepted_mask].iterrows():
    reasons = []
    src = str(row.get("source_span", ""))
    rep = str(row.get("replacement_span", ""))
    chunks = count_diff_chunks(src, rep)
    if chunks >= 2:
        reasons.append(f"multi_concept_edit(chunks={chunks})")
    if leaks_answer(row.get("counterfactual_query", ""), row.get("counterfactual_value", "")):
        reasons.append("leaks_counterfactual_value")
    if leaks_answer(row.get("counterfactual_query", ""), row.get("original_value", "")):
        reasons.append("leaks_original_value")
    flags_by_pair[row["pair_id"]] = "; ".join(reasons)

flagged_ids = {pid for pid, flags in flags_by_pair.items() if flags}
clean_ids = {pid for pid, flags in flags_by_pair.items() if not flags}

print(f"Accepted by validator: {len(flags_by_pair)}")
print(f"Auto-flagged for rejection: {len(flagged_ids)}")
print(f"Passes automated triage: {len(clean_ids)}")
print()
flag_counts = pd.Series([f for f in flags_by_pair.values() if f]).value_counts()
print("Flag breakdown:")
print(flag_counts)

df.loc[df["pair_id"].isin(clean_ids), "approved"] = 1
df.loc[df["pair_id"].isin(flagged_ids), "approved"] = 0
df.loc[df["pair_id"].isin(flagged_ids), "review_notes"] = df.loc[df["pair_id"].isin(flagged_ids), "pair_id"].map(flags_by_pair)
df.loc[df["pair_id"].isin(flags_by_pair.keys()), "reviewer"] = "auto"

df.to_csv(PATH, index=False)
print()
print("Saved. Final approved count:", (df["approved"] == 1).sum())
