#!/usr/bin/env python3
"""
summarize_pairs.py   pair counts by domain, language and attribute family.

Usage:
    python3 summarize_pairs.py data/*.csv                    # print tables
    python3 summarize_pairs.py data/*.csv --latex counts.tex # also write LaTeX tables
    python3 summarize_pairs.py data/file.csv --schema        # print the header only

Only rows that passed screening are counted (screening_status == passed, or approved == 1
in a working file). Use --all to count every row.
"""
import argparse
import pandas as pd

ATTR_ORDER = ["negation", "dosage", "frequency", "subtype", "stage", "temporal", "severity"]


def passed_mask(df):
    if "screening_status" in df.columns:
        return df["screening_status"].astype(str).str.lower().eq("passed")
    if "approved" in df.columns:
        return df["approved"].astype(str).str.lower().isin(["1", "true", "yes"])
    return pd.Series(True, index=df.index)


def tex_table(tab, caption, label):
    cols = "l" + "r" * (tab.shape[1])
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             "Domain & " + " & ".join(map(str, tab.columns)) + r" \\", r"\midrule"]
    for idx, row in tab.iterrows():
        lines.append(str(idx).replace("&", r"\&") + " & " + " & ".join(str(int(v)) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--latex")
    ap.add_argument("--schema", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.schema:
        for f in a.files:
            print(f, list(pd.read_csv(f, nrows=0).columns))
        raise SystemExit
    df = pd.concat([pd.read_csv(f).assign(_file=f) for f in a.files], ignore_index=True)
    if not a.all:
        df = df[passed_mask(df)]
    print(f"{len(df)} pairs")
    by_lang = pd.crosstab(df["domain"], df["language"], margins=True, margins_name="Total")
    attrs = [c for c in ATTR_ORDER if c in set(df["attribute_type"])] + \
            sorted(set(df["attribute_type"]) - set(ATTR_ORDER))
    by_attr = pd.crosstab(df["domain"], df["attribute_type"], margins=True, margins_name="Total")
    by_attr = by_attr[[c for c in attrs if c in by_attr.columns] + ["Total"]]
    print("\nBy domain and language\n", by_lang, sep="")
    print("\nBy domain and attribute family\n", by_attr, sep="")
    if a.latex:
        with open(a.latex, "w") as fh:
            fh.write(tex_table(by_lang, "Screened pairs by domain and query language.", "tab:counts-language") + "\n\n")
            fh.write(tex_table(by_attr, "Screened pairs by domain and attribute family.", "tab:counts-attribute") + "\n")
        print(f"\nwrote {a.latex}")
