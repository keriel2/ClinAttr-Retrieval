#!/usr/bin/env python3
"""
agreement.py

Reads the completed annotation workbooks and the hidden key, and reports
  - agreement between the two annotators on the overlap items (raw agreement, Cohen's kappa),
  - how many pairs each annotator judged acceptable, overall and by group,
  - the list of pair ids that can be used to recompute the retrieval results on the
    human-accepted subset.

Usage:
  python3 agreement.py --a annotation_A_done.xlsx --b annotation_B_done.xlsx \
      --key annotation_key.csv --out-dir annotation_results
  (--b is optional; without it only acceptance rates are reported.)

Definitions
  Answer codes: 1 yes, 0 no, ? cannot judge, blank not answered.
  Acceptable:   Q1 and Q2 answered 1, and every answered Q3 and Q4 is 1 (a ? is ignored).
  With two annotators, an overlap pair is accepted only if both judge it acceptable.
  Intervals are exact (Clopper-Pearson) 95% intervals and treat pairs as independent.
"""
import argparse
from pathlib import Path

import pandas as pd

QCOLS = ["Q1 plausible", "Q2 one change", "Q3 original fits", "Q4 edited fits"]
QS = ["Q1", "Q2", "Q3", "Q4"]


def load(path):
    df = pd.read_excel(path, sheet_name="Items", dtype=str)
    df = df[df["Item"].notna()].copy()
    df["Item"] = df["Item"].str.strip()
    for col, q in zip(QCOLS, QS):
        v = df[col].fillna("").astype(str).str.strip().str.lower()
        v = v.replace({"yes": "1", "y": "1", "no": "0", "n": "0", "1.0": "1", "0.0": "0"})
        df[q] = v.where(v.isin(["1", "0", "?"]), "")
    return df[["Item"] + QS]


def acceptable(row):
    if row["Q1"] != "1" or row["Q2"] != "1":
        return 0
    for q in ("Q3", "Q4"):
        if row[q] == "0":
            return 0
    return 1


def cohen_kappa(x, y):
    n = len(x)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(x, y)) / n
    labels = sorted(set(x) | set(y))
    pe = sum((x.count(l) / n) * (y.count(l) / n) for l in labels)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def exact_ci(k, n):
    if n == 0:
        return float("nan"), float("nan")
    try:
        from scipy.stats import beta
        lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
        hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
        return float(lo), float(hi)
    except Exception:
        z = 1.96
        p = k / n
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        s = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
        return (c - s) / d, (c + s) / d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b")
    ap.add_argument("--key", required=True)
    ap.add_argument("--out-dir", default="annotation_results")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    key = pd.read_csv(args.key)
    A = load(args.a)
    A["acc_A"] = A.apply(acceptable, axis=1)
    unanswered = int((A[QS] == "").any(axis=1).sum())
    print(f"Annotator A: {len(A)} items, {unanswered} with at least one blank answer")

    df = key.merge(A, left_on="item", right_on="Item", how="left")
    df["accepted"] = df["acc_A"]

    agree_rows = []
    if args.b:
        B = load(args.b)
        B["acc_B"] = B.apply(acceptable, axis=1)
        m = A.merge(B, on="Item", suffixes=("_A", "_B"))
        print(f"Overlap: {len(m)} items judged by both")
        for q in QS:
            both = m[(m[q + "_A"].isin(["0", "1"])) & (m[q + "_B"].isin(["0", "1"]))]
            x, y = list(both[q + "_A"]), list(both[q + "_B"])
            agree_rows.append({
                "question": q, "n_both_judged": len(both),
                "raw_agreement": round(sum(a == b for a, b in zip(x, y)) / len(both), 3) if len(both) else float("nan"),
                "cohen_kappa": round(cohen_kappa(x, y), 3),
                "cannot_judge_A": int((m[q + "_A"] == "?").sum()),
                "cannot_judge_B": int((m[q + "_B"] == "?").sum()),
            })
        acc_both = m[["Item", "acc_A", "acc_B"]].copy()
        acc_both["both_accept"] = (acc_both["acc_A"] == 1) & (acc_both["acc_B"] == 1)
        agree_rows.append({
            "question": "acceptable overall", "n_both_judged": len(acc_both),
            "raw_agreement": round((acc_both["acc_A"] == acc_both["acc_B"]).mean(), 3),
            "cohen_kappa": round(cohen_kappa(list(acc_both["acc_A"]), list(acc_both["acc_B"])), 3),
            "cannot_judge_A": "", "cannot_judge_B": ""})
        overlap_both = dict(zip(acc_both["Item"], acc_both["both_accept"].astype(int)))
        df["accepted"] = [overlap_both.get(i, a) for i, a in zip(df["item"], df["acc_A"])]
    agree = pd.DataFrame(agree_rows)
    if len(agree):
        print("\nAgreement on overlap items")
        print(agree.to_string(index=False))
        agree.to_csv(out / "agreement.csv", index=False)

    rows = []
    def add(label, g):
        k, n = int(g["accepted"].sum()), len(g)
        lo, hi = exact_ci(k, n)
        rows.append({"group": label, "n_judged": n, "n_acceptable": k,
                     "rate": round(k / n, 3) if n else float("nan"),
                     "ci95_low": round(lo, 3), "ci95_high": round(hi, 3)})
    judged = df[df["Item"].notna()]
    add("all", judged)
    for col in ["source_label", "language", "generator_model", "attribute_type"]:
        for val, g in judged.groupby(col):
            add(f"{col}={val}", g)
    summary = pd.DataFrame(rows)
    print("\nHuman-judged acceptability of machine-screened pairs")
    print(summary.to_string(index=False))
    summary.to_csv(out / "human_validation_summary.csv", index=False)

    ids = judged[judged["accepted"] == 1][["pair_id", "source_label"]]
    ids.to_csv(out / "human_accepted_pair_ids.csv", index=False)
    print(f"\nsaved {out}/human_validation_summary.csv, human_accepted_pair_ids.csv"
          + (f", agreement.csv" if len(agree) else ""))


if __name__ == "__main__":
    main()
