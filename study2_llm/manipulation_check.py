"""Spike Prompting manipulation check (reviewer-requested robustness check).

Verifies that each spiked story type actually foregrounds its intended
component, using ratings already collected from 25 LLM judging conditions
(80 stories) and 115 human raters (12-story subset).

Spike mapping (generation/methodology.md, Section 2):
    Surreal     -> Novelty    (vocabulary_freshness, plot_uniqueness, surprise)
    Clinical    -> Adherence  (topic_fidelity, tone_fidelity)
    Melancholic -> Resonance  (emotional_impact, empathy, thought_provocation)
    Witty       -> Value      (engagement, stylistic_quality, logical_coherence)

Three tests, matched to construct type:
  1. Winner test (expressive components: Novelty, Resonance): does the matched
     story type score highest on its target sub-dimensions?
  2. Constraint test (Adherence): target held at ceiling (within 0.25 of the
     top group) AND the matched group has the lowest mean over the three
     non-target components ("maximize X, sacrifice whatever necessary").
  3. Human panel: same winner test on the 115 human ratings, where Adherence
     is not at ceiling and can be verified directly.

Value is the global component (operationalized as wit, Appendix D); it is not
expected to separate and no paper claim depends on it. We report its behavior
(top-2 rate) for completeness.

Run: python3 manipulation_check.py
"""
import json
import re
import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw_json"
HUMAN_CSV = ROOT.parent / "study1_human" / "data" / "human_ratings.csv"
OUT = ROOT / "analysis"

SPIKE = {
    "surreal":     ("Novelty",   ["vocabulary_freshness", "plot_uniqueness", "surprise"]),
    "clinical":    ("Adherence", ["topic_fidelity", "tone_fidelity"]),
    "melancholic": ("Resonance", ["emotional_impact", "empathy", "thought_provocation"]),
    "witty":       ("Value",     ["engagement", "stylistic_quality", "logical_coherence"]),
}
TONES = list(SPIKE.keys())
COMPS = [comp for comp, _ in SPIKE.values()]
COMPONENT_DIMS = {comp: dims for comp, dims in SPIKE.values()}
TONE_TO_COMP = {tone: comp for tone, (comp, _) in SPIKE.items()}
COMP_TO_TONE = {comp: tone for tone, comp in TONE_TO_COMP.items()}
CEILING_EPS = 0.25  # "held at ceiling" = within this of the top group

HUMAN_DIMS = {
    "Novelty": ["N_Vocab", "N_Plot", "N_Surprise"],
    "Adherence": ["A_Topic", "A_Tone"],
    "Resonance": ["R_Emotion", "R_Empathy", "R_Thought"],
    "Value": ["V_Engagement", "V_Style", "V_Logic"],
}

# canonical 25 conditions (kept in sync with verify_per_aspect_winners.py)
_verify = (ROOT / "verify_per_aspect_winners.py").read_text()
LLM_FILES = ast.literal_eval(re.search(r"LLM_FILES = (\{.*?\})", _verify, re.S).group(1))


def condition_profile(fname):
    """4x4 DataFrame: mean component score (rows: tone group, cols: component)."""
    data = json.loads((RAW / fname).read_text())
    rows = []
    for r in data["results"]:
        if not r.get("parse_ok", True):
            continue
        tone = r["story_id"].rsplit("_", 1)[-1]
        if tone not in SPIKE:
            continue
        sc = r["scores"]["sub_components"]
        rec = {"tone": tone}
        for comp, dims in COMPONENT_DIMS.items():
            rec[comp] = np.mean([sc[d] for d in dims])
        rows.append(rec)
    return pd.DataFrame(rows).groupby("tone")[COMPS].mean().reindex(TONES)


def human_profile():
    df = pd.read_csv(HUMAN_CSV)
    df["TONE"] = df["TONE"].str.lower().str.strip()
    for comp, dims in HUMAN_DIMS.items():
        df[comp] = df[dims].mean(axis=1)
    return df.groupby("TONE")[COMPS].mean().reindex(TONES)


def main():
    profiles = {cond: condition_profile(f) for cond, f in LLM_FILES.items()}
    n = len(profiles)

    # ---- Test 1: winner test, expressive components ----
    print("=== Test 1: winner test (matched type scores highest on target dims) ===")
    winner_hits = {c: 0 for c in COMPS}
    deltas = {c: [] for c in COMPS}
    for g in profiles.values():
        for comp in COMPS:
            tone = COMP_TO_TONE[comp]
            winner_hits[comp] += g[comp].idxmax() == tone
            deltas[comp].append(g.loc[tone, comp] - g.drop(tone)[comp].mean())
    for comp in COMPS:
        print(f"  {comp:<10} hits {winner_hits[comp]:>2}/{n}  mean delta (matched - others) {np.mean(deltas[comp]):+.2f}")
    print("  NOTE: Adherence winner test has no resolution: all groups at ceiling"
          " (see Test 2); its hit count is tie-break noise.")

    # ---- Test 2: constraint test for Adherence ----
    print("\n=== Test 2: constraint test for Adherence (clinical group) ===")
    ceiling = sacrifice = both = res_lowest = 0
    adh_range = []
    for g in profiles.values():
        c_ceiling = g.loc["clinical", "Adherence"] >= g["Adherence"].max() - CEILING_EPS
        nontarget = g[["Novelty", "Resonance", "Value"]].mean(axis=1)
        c_sac = nontarget.idxmin() == "clinical"
        ceiling += c_ceiling
        sacrifice += c_sac
        both += c_ceiling and c_sac
        res_lowest += g["Resonance"].idxmin() == "clinical"
        adh_range.append((g["Adherence"].min(), g["Adherence"].max()))
    print(f"  Adherence held at ceiling (within {CEILING_EPS} of top group): {ceiling}/{n}")
    print(f"  clinical = lowest mean over 3 non-target components:          {sacrifice}/{n}")
    print(f"  both together:                                                {both}/{n}")
    print(f"  clinical = lowest-Resonance group:                            {res_lowest}/{n}")

    # ---- Value behavior (global component; completeness only) ----
    print("\n=== Value (global component, operationalized as wit; no claim rests on it) ===")
    v_rank1 = v_top2 = 0
    for g in profiles.values():
        rank = int(g["Value"].rank(ascending=False)["witty"])
        v_rank1 += rank == 1
        v_top2 += rank <= 2
    print(f"  witty rank 1 on Value: {v_rank1}/{n} | top-2: {v_top2}/{n}")

    # ---- Test 3: human panel ----
    print("\n=== Test 3: human panel (115 ratings, 12 stories) ===")
    hg = human_profile()
    print(hg.round(2).to_string())
    for comp in COMPS:
        tone = COMP_TO_TONE[comp]
        delta = hg.loc[tone, comp] - hg.drop(tone)[comp].mean()
        hit = hg[comp].idxmax() == tone
        print(f"  {comp:<10} matched {hg.loc[tone, comp]:.2f} vs others"
              f" {hg.drop(tone)[comp].mean():.2f} (delta {delta:+.2f}) hit={hit}")
    h_nontarget = hg[["Novelty", "Resonance", "Value"]].mean(axis=1)
    print(f"  clinical lowest non-target mean (humans): {h_nontarget.idxmin() == 'clinical'}"
          f" ({h_nontarget['clinical']:.2f}, others {h_nontarget.drop('clinical').mean():.2f})")
    h_sd = hg.std(axis=1)
    print(f"  most even profile (lowest across-component SD): {h_sd.idxmin()}"
          f" (witty SD {h_sd['witty']:.2f})")

    # ---- pooled LLM profile ----
    pooled = sum(profiles.values()) / n
    print("\n=== Pooled LLM component profile by spike group ===")
    print(pooled.round(2).to_string())

    # ---- persist ----
    per_rows = []
    for cond, g in profiles.items():
        for comp in COMPS:
            tone = COMP_TO_TONE[comp]
            per_rows.append({
                "condition": cond, "component": comp,
                "matched": g.loc[tone, comp],
                "others": g.drop(tone)[comp].mean(),
                "winner_hit": g[comp].idxmax() == tone,
            })
    pd.DataFrame(per_rows).to_csv(OUT / "manipulation_check_percondition.csv", index=False)
    pooled.to_csv(OUT / "manipulation_check_profile.csv")
    hg.to_csv(OUT / "manipulation_check_human.csv")
    print(f"\nSaved: {OUT/'manipulation_check_percondition.csv'}")
    print(f"Saved: {OUT/'manipulation_check_profile.csv'}")
    print(f"Saved: {OUT/'manipulation_check_human.csv'}")


if __name__ == "__main__":
    main()
