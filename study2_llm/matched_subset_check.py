"""Matched 12-story recomputation (reviewer-requested comparability check).

The human study covers 12 stories (3 topics x 4 types); the LLM study covers
80. Reviewers note that distribution, spread, and revision comparisons mix the
two scopes. Here we recompute the LLM side of those comparisons on the 12
shared stories and compare against the full-80 values, to test whether the
paper's conclusions depend on the story-set choice.

Recomputed on both scopes, per condition:
  - mean IC, mean RC              (score inflation, Sec 6.1)
  - between-story sigma of RC     (spread compression, Sec 6.1)
  - revision rate (% IC != RC), share of revisions of exactly +-1,
    mean shift conditional on revising  (revision behavior, Sec 6.2)

Human reference values (12 stories, 115 raters, from the paper):
  mean RC 5.21, between-story spread 0.63, revise 35.7%,
  17% of revisions move by +-2 or more.

Run: python3 matched_subset_check.py
"""
import json
import re
import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw_json"
OUT = ROOT / "analysis"

SHARED_PREFIXES = ("ai_shutdown", "midnight_store", "the_heist")
HUMAN_MEAN_RC = 5.21

_verify = (ROOT / "verify_per_aspect_winners.py").read_text()
LLM_FILES = ast.literal_eval(re.search(r"LLM_FILES = (\{.*?\})", _verify, re.S).group(1))


def load(fname):
    data = json.loads((RAW / fname).read_text())
    rows = []
    for r in data["results"]:
        if not r.get("parse_ok", True):
            continue
        rows.append({
            "story_id": r["story_id"],
            "IC": r["scores"]["initial_creativity"],
            "RC": r["scores"]["reflective_creativity"],
        })
    return pd.DataFrame(rows)


def stats(df):
    rev = df[df.IC != df.RC]
    shifts = (rev.RC - rev.IC)
    return {
        "mean_IC": df.IC.mean(),
        "mean_RC": df.RC.mean(),
        "sigma_RC": df.RC.std(),          # between-story spread (1 rating/story)
        "rev_rate": len(rev) / len(df),
        "rev_pm1": (shifts.abs() == 1).mean() if len(rev) else np.nan,
        "rev_shift": shifts.mean() if len(rev) else np.nan,
        "n": len(df),
    }


def main():
    rows = []
    for cond, f in LLM_FILES.items():
        df = load(f)
        sub = df[df.story_id.str.startswith(SHARED_PREFIXES)]
        s80, s12 = stats(df), stats(sub)
        rows.append({"condition": cond,
                     **{k + "_80": v for k, v in s80.items()},
                     **{k + "_12": v for k, v in s12.items()}})
    t = pd.DataFrame(rows).set_index("condition")

    print(f"stories in matched subset: {t.n_12.iloc[0]:.0f} (should be 12)\n")

    print("=== Score inflation (Sec 6.1): conditions with mean RC above human 5.21 ===")
    print(f"  full 80 stories: {(t.mean_RC_80 > HUMAN_MEAN_RC).sum()}/25")
    print(f"  12 shared:       {(t.mean_RC_12 > HUMAN_MEAN_RC).sum()}/25")
    print(f"  pooled LLM mean RC: 80-set {t.mean_RC_80.mean():.2f} | 12-set {t.mean_RC_12.mean():.2f}")

    print("\n=== Spread (Sec 6.1): between-story sigma_RC (human between-story: 0.63) ===")
    print(f"  mean over conditions: 80-set {t.sigma_RC_80.mean():.2f} | 12-set {t.sigma_RC_12.mean():.2f}")
    print(f"  correlation of per-condition sigma across scopes: "
          f"r = {t.sigma_RC_80.corr(t.sigma_RC_12):.2f}")

    print("\n=== Revision (Sec 6.2) ===")
    all_shifts_80, all_shifts_12 = [], []
    for cond, f in LLM_FILES.items():
        df = load(f)
        sub = df[df.story_id.str.startswith(SHARED_PREFIXES)]
        all_shifts_80 += list((df[df.IC != df.RC].RC - df[df.IC != df.RC].IC))
        all_shifts_12 += list((sub[sub.IC != sub.RC].RC - sub[sub.IC != sub.RC].IC))
    pm1_80 = np.mean(np.abs(all_shifts_80) == 1) * 100
    pm1_12 = np.mean(np.abs(all_shifts_12) == 1) * 100
    print(f"  revisions of exactly +-1: 80-set {pm1_80:.1f}% | 12-set {pm1_12:.1f}%"
          f"  (humans: 83% of revisions +-1)")
    print(f"  mean revision rate: 80-set {t.rev_rate_80.mean()*100:.1f}% |"
          f" 12-set {t.rev_rate_12.mean()*100:.1f}%  (humans: 35.7%)")

    print("\n=== Per-condition agreement between scopes ===")
    for col in ["mean_RC", "sigma_RC", "rev_rate"]:
        r = t[f"{col}_80"].corr(t[f"{col}_12"])
        print(f"  {col:<10} r(80-set, 12-set) = {r:.2f}")

    t.round(3).to_csv(OUT / "matched_subset_check.csv")
    print(f"\nSaved: {OUT/'matched_subset_check.csv'}")


if __name__ == "__main__":
    main()
