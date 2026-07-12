"""Bootstrap stability check for beta vectors and dR2 (reviewer-requested robustness check).

Reviewer comment: "The stability of core statistics such as beta vectors and dR2 is
questionable" given 115 human ratings and one sample per LLM condition.

We bootstrap what we have:
  1. Human side: resample the 115 ratings with replacement (B times),
     recompute the standardized beta vector (RC ~ 11 subs) and dR2.
     Report: 95% CI per weight, how often the top-3 weights stay top-3,
     cosine of each bootstrap beta to the full-sample beta.
  2. LLM side: for each condition, resample its 80 stories with replacement,
     recompute dR2 and cosine-to-human. Report 95% CIs and whether the
     paper's grouping (high vs low dR2 conditions) is preserved.

Run: python3 bootstrap_stability.py
"""
import json
import re
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

RNG = np.random.default_rng(42)
B = 10_000
B_LLM = 2_000  # per condition; 25 conditions

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw_json"
HUMAN_CSV = ROOT.parent / "study1_human" / "data" / "human_ratings.csv"
OUT = ROOT / "analysis"

SUBS = ["emotional_impact", "topic_fidelity", "vocabulary_freshness",
        "plot_uniqueness", "surprise", "empathy", "thought_provocation",
        "engagement", "stylistic_quality", "logical_coherence", "tone_fidelity"]
HUMAN_COLS = {"R_Emotion": "emotional_impact", "A_Topic": "topic_fidelity",
              "N_Vocab": "vocabulary_freshness", "N_Plot": "plot_uniqueness",
              "N_Surprise": "surprise", "R_Empathy": "empathy",
              "R_Thought": "thought_provocation", "V_Engagement": "engagement",
              "V_Style": "stylistic_quality", "V_Logic": "logical_coherence",
              "A_Tone": "tone_fidelity"}

_verify = (ROOT / "verify_per_aspect_winners.py").read_text()
LLM_FILES = ast.literal_eval(re.search(r"LLM_FILES = (\{.*?\})", _verify, re.S).group(1))


def zbeta(d, y, xcols):
    kept = [c for c in xcols if d[c].std() > 0]
    Z = (d[kept] - d[kept].mean()) / d[kept].std()
    yz = (d[y] - d[y].mean()) / d[y].std()
    beta = pd.Series(0.0, index=xcols)
    beta[kept] = LinearRegression().fit(Z, yz).coef_
    return beta.values


def dr2(d, y, xcols):
    r_ic = LinearRegression().fit(d[["IC"]], d[y]).score(d[["IC"]], d[y])
    Xf = d[["IC"] + xcols]
    r_full = LinearRegression().fit(Xf, d[y]).score(Xf, d[y])
    return r_full - r_ic


def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


def load_llm(fname):
    data = json.loads((RAW / fname).read_text())
    rows = []
    for r in data["results"]:
        if not r.get("parse_ok", True):
            continue
        rec = {"IC": r["scores"]["initial_creativity"],
               "RC": r["scores"]["reflective_creativity"]}
        rec.update(r["scores"]["sub_components"])
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    # ---------- 1. human side ----------
    hf = (pd.read_csv(HUMAN_CSV).rename(columns=HUMAN_COLS)
          .rename(columns={"O_Final_Creativity": "RC", "O_Initial_Creativity": "IC"}))
    hd = hf[["RC", "IC"] + SUBS].dropna().reset_index(drop=True)
    full_beta = zbeta(hd, "RC", SUBS)
    full_top3 = set(np.argsort(full_beta)[-3:])
    full_dr2 = dr2(hd, "RC", SUBS)

    betas = np.empty((B, len(SUBS)))
    dr2s = np.empty(B)
    top3_kept = 0
    cos_to_full = np.empty(B)
    n = len(hd)
    for b in range(B):
        s = hd.iloc[RNG.integers(0, n, n)]
        bb = zbeta(s, "RC", SUBS)
        betas[b] = bb
        dr2s[b] = dr2(s, "RC", SUBS)
        cos_to_full[b] = cosine(bb, full_beta)
        top3_kept += set(np.argsort(bb)[-3:]) == full_top3

    print(f"=== Human bootstrap (B={B}, n={n} ratings resampled) ===")
    lo, hi = np.percentile(betas, [2.5, 97.5], axis=0)
    tab = pd.DataFrame({"beta": full_beta, "CI_lo": lo, "CI_hi": hi}, index=SUBS)
    print(tab.sort_values("beta", ascending=False).round(3).to_string())
    print(f"\ntop-3 weights identical to full sample: {100*top3_kept/B:.1f}% of resamples")
    print(f"cosine(bootstrap beta, full beta): mean {cos_to_full.mean():.3f}, "
          f"5th pct {np.percentile(cos_to_full, 5):.3f}")
    print(f"human dR2 = {full_dr2:.3f}, 95% CI [{np.percentile(dr2s, 2.5):.3f}, "
          f"{np.percentile(dr2s, 97.5):.3f}]")

    # ---------- 2. LLM side ----------
    print(f"\n=== LLM bootstrap (B={B_LLM} per condition, 80 stories resampled) ===")
    rows = []
    for cond, f in LLM_FILES.items():
        d = load_llm(f).reset_index(drop=True)
        point_dr2 = dr2(d, "RC", SUBS)
        point_cos = cosine(zbeta(d, "RC", SUBS), full_beta)
        bs_dr2 = np.full(B_LLM, np.nan)
        bs_cos = np.full(B_LLM, np.nan)
        degenerate = 0
        for b in range(B_LLM):
            s = d.iloc[RNG.integers(0, len(d), len(d))]
            if s["RC"].std() == 0:  # all-ceiling resample; stats undefined
                degenerate += 1
                continue
            bs_dr2[b] = dr2(s, "RC", SUBS)
            bs_cos[b] = cosine(zbeta(s, "RC", SUBS), full_beta)
        rows.append({"condition": cond, "dR2": point_dr2,
                     "dR2_lo": np.nanpercentile(bs_dr2, 2.5),
                     "dR2_hi": np.nanpercentile(bs_dr2, 97.5),
                     "cos": point_cos,
                     "cos_lo": np.nanpercentile(bs_cos, 2.5),
                     "cos_hi": np.nanpercentile(bs_cos, 97.5),
                     "degen_pct": 100 * degenerate / B_LLM})
    t = pd.DataFrame(rows).set_index("condition").round(3)
    print(t.to_string())

    # stability of the paper's qualitative grouping
    hi_group = t[t.dR2 >= 0.25].index
    lo_group = t[t.dR2 <= 0.05].index
    sep = (t.loc[hi_group, "dR2_lo"].min() if len(hi_group) else np.nan,
           t.loc[lo_group, "dR2_hi"].max() if len(lo_group) else np.nan)
    print(f"\nhigh-dR2 group lower CI bound: {sep[0]:.3f} | "
          f"low-dR2 group upper CI bound: {sep[1]:.3f} "
          f"({'no overlap' if sep[0] > sep[1] else 'overlap'})")

    t.to_csv(OUT / "bootstrap_stability_llm.csv")
    tab.round(3).to_csv(OUT / "bootstrap_stability_human.csv")
    print(f"\nSaved: {OUT/'bootstrap_stability_llm.csv'}")
    print(f"Saved: {OUT/'bootstrap_stability_human.csv'}")


if __name__ == "__main__":
    main()
