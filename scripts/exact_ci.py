#!/usr/bin/env python3
"""
exact_ci.py   exact (Clopper-Pearson) interval and exact two-sided binomial test.

Usage:
    python3 exact_ci.py --k 5 --n 29                 # 95% interval, test against .25
    python3 exact_ci.py --k 5 --n 29 --p0 0.5 --alpha 0.05

Pairs are treated as independent. The test is descriptive and uncorrected for multiple
comparisons.
"""
import argparse
from scipy.stats import beta, binomtest


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True, help="successes (pairs consistent)")
    ap.add_argument("--n", type=int, required=True, help="pairs")
    ap.add_argument("--p0", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    lo, hi = clopper_pearson(a.k, a.n, a.alpha)
    p = binomtest(a.k, a.n, a.p0, alternative="two-sided").pvalue
    print(f"k={a.k} n={a.n} rate={a.k / a.n:.3f}  exact {100 * (1 - a.alpha):.0f}% interval "
          f"[{lo:.3f}, {hi:.3f}]  exact binomial p vs {a.p0}: {p:.3f}")
