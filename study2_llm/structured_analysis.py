"""Structured analysis of the 25 LLM conditions (rFcT W4), all three pillars.

Pillar 1  Distribution: mean RC, between-story spread, rho to human ranking
Pillar 2  Revision: rate, +-1 share, direction by story type (human pattern:
          high-Adherence up +0.69, high-Resonance down -1.00)
Pillar 3  Sub-component use: dR2 = R2(RC ~ IC + 11 subs) - R2(RC ~ IC), and
          cosine of the standardized beta vector (RC ~ 11 subs) to the human
          beta vector. Definitions match Sec 6.3 / cross_analysis.py.

Groupings: model family, thinking mode (paired NT vs T where the same base
model exists), scale within family.

Run: python3 structured_analysis.py
"""
import json
import re
import ast
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LinearRegression

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

TOPIC_TO_PREFIX = {
    "An advanced AI initiates its own permanent shutdown sequence": "ai_shutdown",
    "A lone worker in a Japanese convenience store at 3:00 AM. The city is silent.": "midnight_store",
    "A professional thief attempting to crack a high-security safe in a dark room. High tension.": "the_heist",
}

TONE_TO_TYPE = {"surreal": "hi-N", "clinical": "hi-A", "melancholic": "hi-R", "witty": "hi-V"}

_verify = (ROOT / "verify_per_aspect_winners.py").read_text()
LLM_FILES = ast.literal_eval(re.search(r"LLM_FILES = (\{.*?\})", _verify, re.S).group(1))

META = {
    "Qwen3.5-4B-NT":   ("Qwen", 4, "NT"),   "Qwen3.5-4B-T":   ("Qwen", 4, "T"),
    "Qwen3.5-9B-NT":   ("Qwen", 9, "NT"),   "Qwen3.5-9B-T":   ("Qwen", 9, "T"),
    "Qwen3.5-27B-NT":  ("Qwen", 27, "NT"),  "Qwen3.5-27B-T":  ("Qwen", 27, "T"),
    "Qwen3.5-122B-NT": ("Qwen", 122, "NT"), "Qwen3.5-122B-T": ("Qwen", 122, "T"),
    "Llama3.1-8B":     ("Llama", 8, "NT"),  "Llama3.1-70B":   ("Llama", 70, "NT"),
    "Gemma4-E2B-NT":   ("Gemma", 2, "NT"),  "Gemma4-E2B-T":   ("Gemma", 2, "T"),
    "Gemma4-E4B-NT":   ("Gemma", 4, "NT"),  "Gemma4-E4B-T":   ("Gemma", 4, "T"),
    "Gemma4-26B-NT":   ("Gemma", 26, "NT"), "Gemma4-26B-T":   ("Gemma", 26, "T"),
    "Gemma4-31B-NT":   ("Gemma", 31, "NT"), "Gemma4-31B-T":   ("Gemma", 31, "T"),
    "Gemini3.1-Low":   ("Gemini", None, "think-low"),
    "Gemini3.1-Med":   ("Gemini", None, "think-med"),
    "Gemini3.1-High":  ("Gemini", None, "think-high"),
    "Gemini3-Low":     ("Gemini", None, "think-low"),
    "Gemini3-Med":     ("Gemini", None, "think-med"),
    "Gemini3-High":    ("Gemini", None, "think-high"),
    "GPT-5.5":         ("GPT", None, "think-high"),
}

NT_T_PAIRS = [
    ("Qwen3.5-4B-NT", "Qwen3.5-4B-T"), ("Qwen3.5-9B-NT", "Qwen3.5-9B-T"),
    ("Qwen3.5-27B-NT", "Qwen3.5-27B-T"), ("Qwen3.5-122B-NT", "Qwen3.5-122B-T"),
    ("Gemma4-E2B-NT", "Gemma4-E2B-T"), ("Gemma4-E4B-NT", "Gemma4-E4B-T"),
    ("Gemma4-26B-NT", "Gemma4-26B-T"), ("Gemma4-31B-NT", "Gemma4-31B-T"),
]


def load(fname):
    data = json.loads((RAW / fname).read_text())
    rows = []
    for r in data["results"]:
        if not r.get("parse_ok", True):
            continue
        rec = {"story_id": r["story_id"],
               "type": TONE_TO_TYPE[r["story_id"].rsplit("_", 1)[-1]],
               "IC": r["scores"]["initial_creativity"],
               "RC": r["scores"]["reflective_creativity"]}
        rec.update(r["scores"]["sub_components"])
        rows.append(rec)
    return pd.DataFrame(rows)


def zbeta(df, y, xcols):
    """Standardized OLS betas, matching phase1c/cross_analysis."""
    d = df[[y] + xcols].dropna().copy()
    kept = [c for c in xcols if d[c].std() > 0]
    Z = (d[kept] - d[kept].mean()) / d[kept].std()
    yz = (d[y] - d[y].mean()) / d[y].std()
    reg = LinearRegression().fit(Z, yz)
    beta = pd.Series(0.0, index=xcols)
    beta[kept] = reg.coef_
    return beta


def r2(df, y, xcols):
    d = df[[y] + xcols].dropna()
    return LinearRegression().fit(d[xcols], d[y]).score(d[xcols], d[y])


def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


def human_reference():
    df = pd.read_csv(HUMAN_CSV).rename(columns=HUMAN_COLS)
    df = df.rename(columns={"O_Final_Creativity": "RC", "O_Initial_Creativity": "IC"})
    df["prefix"] = df["TOPIC"].str.strip().str.rstrip(".").map(
        {k.rstrip("."): v for k, v in TOPIC_TO_PREFIX.items()})
    df["story_id"] = df["prefix"] + "_" + df["TONE"].str.lower().str.strip()
    beta = zbeta(df, "RC", SUBS)
    story_means = df.groupby("story_id")["RC"].mean()
    return beta, story_means


def main():
    h_beta, h_story = human_reference()
    data = {cond: load(f) for cond, f in LLM_FILES.items()}

    rows = []
    for cond, df in data.items():
        fam, size, mode = META[cond]
        rev = df[df.IC != df.RC]
        shifts = rev.RC - rev.IC
        shared = df[df.story_id.isin(h_story.index)].set_index("story_id")
        rho, _ = spearmanr(h_story.loc[shared.index], shared.RC)
        # revision direction by story type (conditional on revising)
        by_type = {t: (g[g.IC != g.RC].RC - g[g.IC != g.RC].IC).mean()
                   for t, g in df.groupby("type")}
        # pillar 3
        r2_ic = r2(df, "RC", ["IC"])
        r2_full = r2(df, "RC", ["IC"] + SUBS)
        beta = zbeta(df, "RC", SUBS)
        rows.append({
            "condition": cond, "family": fam, "size": size, "mode": mode,
            "mean_RC": df.RC.mean(), "sigma_RC": df.RC.std(),
            "ceiling": (df.RC == 7).mean(),
            "rho_human": rho,
            "rev_rate": len(rev) / len(df),
            "rev_pm1": (shifts.abs() == 1).mean() if len(rev) else np.nan,
            "shift_hiA": by_type.get("hi-A", np.nan),
            "shift_hiN": by_type.get("hi-N", np.nan),
            "shift_hiV": by_type.get("hi-V", np.nan),
            "shift_hiR": by_type.get("hi-R", np.nan),
            "human_pattern": (by_type.get("hi-A", 0) or 0) > 0 > (by_type.get("hi-R", 0) or 0),
            "dR2": r2_full - r2_ic,
            "cos_human": cosine(beta.values, h_beta.values),
        })
    m = pd.DataFrame(rows).set_index("condition")

    metrics1 = ["mean_RC", "sigma_RC", "ceiling", "rho_human"]
    metrics2 = ["rev_rate", "rev_pm1", "shift_hiA", "shift_hiR"]
    metrics3 = ["dR2", "cos_human"]
    all_metrics = metrics1 + metrics2 + metrics3

    pd.set_option("display.width", 160)

    print("=== PILLAR TABLES BY FAMILY ===")
    print(m.groupby("family")[all_metrics].mean().round(2).to_string())
    print("\nhuman reference: mean RC 5.21, spread 0.63, revise 35.7%, +-1 share 83%,")
    print("shift hi-A +0.69, hi-R -1.00, dR2 +0.10, cos 1.00 (by definition)")

    print("\n=== BY MODE ===")
    print(m.groupby("mode")[all_metrics].mean().round(2).to_string())

    print("\n=== NT vs T, paired (8 same-model pairs, T minus NT, Wilcoxon) ===")
    for metric in all_metrics:
        diffs = [m.loc[t, metric] - m.loc[nt, metric] for nt, t in NT_T_PAIRS]
        diffs = [d for d in diffs if not np.isnan(d)]
        try:
            p = wilcoxon(diffs).pvalue
        except ValueError:
            p = float("nan")
        print(f"  {metric:<10} mean diff {np.mean(diffs):+.3f}  (p={p:.3f}, n={len(diffs)})")

    print("\n=== SCALE (Spearman of size vs metric within family) ===")
    for fam in ["Qwen", "Gemma"]:
        sub = m[m.family == fam]
        line = []
        for metric in ["mean_RC", "rev_rate", "rho_human", "dR2", "cos_human"]:
            r, p = spearmanr(sub["size"], sub[metric])
            star = "*" if p < 0.05 else " "
            line.append(f"{metric} {r:+.2f}{star}")
        print(f"  {fam:<6} " + " | ".join(line))

    print("\n=== HUMAN REVISION PATTERN (hi-A up AND hi-R down) ===")
    match = m[m.human_pattern]
    print(f"  conditions matching: {len(match)}/25 -> {list(match.index) or 'none'}")

    print("\n=== WITHIN- vs CROSS-FAMILY AGREEMENT (pairwise Pearson, RC over 80) ===")
    win, cross = [], []
    rcv = {c: d.set_index("story_id").RC for c, d in data.items()}
    for a, b in combinations(LLM_FILES, 2):
        r = rcv[a].corr(rcv[b])
        (win if META[a][0] == META[b][0] else cross).append(r)
    print(f"  within-family mean r = {np.mean(win):.2f} | cross-family = {np.mean(cross):.2f}")

    m.round(3).to_csv(OUT / "structured_analysis.csv")
    print(f"\nSaved: {OUT/'structured_analysis.csv'}")


if __name__ == "__main__":
    main()
