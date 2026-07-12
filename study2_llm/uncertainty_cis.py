"""Bootstrap CIs for the secondary statistics (upyP Suggestions 7 and 8).

Computes 95% percentile bootstrap confidence intervals for:
  1. Human revision-by-story-type: revision rate and mean shift conditional
     on revising, per type (resampling raters within type, 10,000 reps).
  2. Human-LLM ranking correlations on the 12 shared stories: per-condition
     Spearman rho between condition RC and human consensus RC, plus the mean
     rho across the 25 conditions (resampling stories, 2,000 reps).
  3. Per-sub-component best-matching condition (Appendix winners table):
     CI for each component's top rho (resampling stories, 2,000 reps).
  4. Within-story SD from the 10-repeat rerun files (resampling stories,
     10,000 reps).

Run: python3 study2_llm/uncertainty_cis.py
Outputs: analysis/uncertainty_cis_*.csv
"""
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw_json"
OUT = ROOT / "analysis"
HUMAN_CSV = ROOT.parent / "study1_human" / "data" / "human_ratings.csv"
SENS = ROOT / "temp_sensitivity" / "results"

RNG = np.random.default_rng(13010)

TOPIC_MAP = {
    "An advanced AI initiates its own permanent shutdown sequence": "ai_shutdown",
    "A lone worker in a Japanese convenience store at 3:00 AM. The city is silent.": "midnight_store",
    "A professional thief attempting to crack a high-security safe in a dark room. High tension.": "the_heist",
}
SUB_MAP = {
    "R_Emotion": "emotional_impact", "A_Topic": "topic_fidelity",
    "N_Vocab": "vocabulary_freshness", "N_Plot": "plot_uniqueness",
    "N_Surprise": "surprise", "R_Empathy": "empathy",
    "R_Thought": "thought_provocation", "V_Engagement": "engagement",
    "V_Style": "stylistic_quality", "V_Logic": "logical_coherence",
    "A_Tone": "tone_fidelity",
}

_verify = (ROOT / "verify_per_aspect_winners.py").read_text()
LLM_FILES = ast.literal_eval(re.search(r"LLM_FILES = (\{.*?\})", _verify, re.S).group(1))

TEN_REPEAT_FILES = {
    "GPT-5.5 (high)": "sampling_gpt55_high_default.json",
    "Gemini 3.1 Pro (low)": "sampling_gemini31_low_t1.json",
    "Llama 3.1 70B (NT)": "temp_sens_llama31_70b_nothink.json",
    "Qwen 3.5 4B (NT)": "temp_sens_qwen35_4b_nothink.json",
}


def pct_ci(samples, lo=2.5, hi=97.5):
    s = np.asarray(samples, dtype=float)
    s = s[~np.isnan(s)]
    return np.percentile(s, lo), np.percentile(s, hi)


def load_condition(fname):
    """Raw judge file -> one row per story (dedup keeps the latest record)."""
    data = json.loads((RAW / fname).read_text())
    rows = {}
    for r in data["results"]:
        if not r.get("parse_ok", True):
            continue
        rows[r["story_id"]] = {"story_id": r["story_id"],
                               "RC": r["scores"]["reflective_creativity"],
                               **r["scores"]["sub_components"]}
    return pd.DataFrame(rows.values())


def human_revision_cis(h, n_boot=10_000):
    print("== 1. Human revision by story type (95% bootstrap CIs) ==")
    rows = []
    for tone, g in h.groupby("TONE"):
        ic, rc = g.O_Initial_Creativity.values, g.O_Final_Creativity.values
        n = len(g)
        revised = ic != rc
        rate = revised.mean()
        shift = (rc - ic)[revised].mean()
        rates, shifts = [], []
        for _ in range(n_boot):
            idx = RNG.integers(0, n, n)
            rv = ic[idx] != rc[idx]
            rates.append(rv.mean())
            shifts.append((rc[idx] - ic[idx])[rv].mean() if rv.any() else np.nan)
        r_lo, r_hi = pct_ci(rates)
        s_lo, s_hi = pct_ci(shifts)
        rows.append({"type": tone, "n_raters": n,
                     "revise_pct": 100 * rate, "revise_lo": 100 * r_lo, "revise_hi": 100 * r_hi,
                     "shift": shift, "shift_lo": s_lo, "shift_hi": s_hi})
        print(f"  {tone:12s} n={n:3d}  revise {100*rate:4.1f}% [{100*r_lo:.1f}, {100*r_hi:.1f}]"
              f"  shift {shift:+.2f} [{s_lo:+.2f}, {s_hi:+.2f}]")
    return pd.DataFrame(rows)


def ranking_cis(h, n_boot=2_000):
    print("\n== 2. Human-LLM RC ranking correlation, 12 shared stories ==")
    h = h.copy()
    h["story_id"] = h.TOPIC.map(TOPIC_MAP) + "_" + h.TONE.str.lower()
    consensus = h.groupby("story_id").O_Final_Creativity.mean()
    stories = consensus.index.to_numpy()
    k = len(stories)

    cond_rc = {}
    for label, fname in LLM_FILES.items():
        df = load_condition(fname).set_index("story_id")
        cond_rc[label] = df.RC.reindex(stories)

    rows, boot_means = [], np.zeros((n_boot, len(cond_rc)))
    boot_idx = RNG.integers(0, k, (n_boot, k))
    hvals = consensus.values
    for j, (label, rc) in enumerate(cond_rc.items()):
        rho = spearmanr(hvals, rc.values).statistic
        boots = []
        for b in range(n_boot):
            idx = boot_idx[b]
            hv, lv = hvals[idx], rc.values[idx]
            if len(set(hv)) < 2 or len(set(lv)) < 2:
                boots.append(np.nan)
            else:
                boots.append(spearmanr(hv, lv).statistic)
        boot_means[:, j] = boots
        lo, hi = pct_ci(boots)
        rows.append({"condition": label, "rho": rho, "lo": lo, "hi": hi})
    t = pd.DataFrame(rows).sort_values("rho", ascending=False)

    mean_rho = t.rho.mean()
    mean_boots = np.nanmean(boot_means, axis=1)
    m_lo, m_hi = pct_ci(mean_boots)
    print(f"  mean rho across 25 conditions: {mean_rho:+.2f} [{m_lo:+.2f}, {m_hi:+.2f}]")
    print(f"  best : {t.iloc[0].condition}  rho {t.iloc[0].rho:+.2f} [{t.iloc[0].lo:+.2f}, {t.iloc[0].hi:+.2f}]")
    print(f"  worst: {t.iloc[-1].condition}  rho {t.iloc[-1].rho:+.2f} [{t.iloc[-1].lo:+.2f}, {t.iloc[-1].hi:+.2f}]")
    t.loc[len(t)] = {"condition": "MEAN_ACROSS_25", "rho": mean_rho, "lo": m_lo, "hi": m_hi}
    return t


def subcomponent_winner_cis(h, n_boot=2_000):
    print("\n== 3. Per-sub-component best condition (winners table), 12 shared stories ==")
    h = h.copy()
    h["story_id"] = h.TOPIC.map(TOPIC_MAP) + "_" + h.TONE.str.lower()
    rows = []
    conds = {label: load_condition(fname).set_index("story_id")
             for label, fname in LLM_FILES.items()}
    for hcol, sub in SUB_MAP.items():
        consensus = h.groupby("story_id")[hcol].mean()
        stories = consensus.index.to_numpy()
        hvals = consensus.values
        best = (None, -2.0, None)
        for label, df in conds.items():
            lv = df[sub].reindex(stories).values
            rho = spearmanr(hvals, lv).statistic
            if not np.isnan(rho) and rho > best[1]:
                best = (label, rho, lv)
        label, rho, lv = best
        boots = []
        for _ in range(n_boot):
            idx = RNG.integers(0, len(stories), len(stories))
            hv, lb = hvals[idx], lv[idx]
            if len(set(hv)) < 2 or len(set(lb)) < 2:
                boots.append(np.nan)
            else:
                boots.append(spearmanr(hv, lb).statistic)
        lo, hi = pct_ci(boots)
        rows.append({"sub_component": sub, "best_condition": label,
                     "rho": rho, "lo": lo, "hi": hi})
        print(f"  {sub:22s} {label:28s} rho {rho:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    return pd.DataFrame(rows)


def rerun_sd_cis(n_boot=10_000):
    print("\n== 4. Within-story SD, 10-repeat runs (95% bootstrap CIs) ==")
    rows = []
    for label, fname in TEN_REPEAT_FILES.items():
        data = json.loads((SENS / fname).read_text())
        per_story = {}
        for r in data["results"]:
            if r.get("parse_ok"):
                per_story.setdefault(r["story_id"], []).append(
                    r["scores"]["reflective_creativity"])
        sds = np.array([np.std(v, ddof=1) for v in per_story.values() if len(v) >= 2])
        mean_sd = sds.mean()
        boots = [sds[RNG.integers(0, len(sds), len(sds))].mean() for _ in range(n_boot)]
        lo, hi = pct_ci(boots)
        rows.append({"judge": label, "n_stories": len(sds),
                     "mean_within_story_sd": mean_sd, "lo": lo, "hi": hi})
        print(f"  {label:22s} SD {mean_sd:.3f} [{lo:.3f}, {hi:.3f}]  ({len(sds)} stories)")
    return pd.DataFrame(rows)


def main():
    h = pd.read_csv(HUMAN_CSV)
    OUT.mkdir(exist_ok=True)
    human_revision_cis(h).round(4).to_csv(OUT / "uncertainty_cis_human_revision.csv", index=False)
    ranking_cis(h).round(4).to_csv(OUT / "uncertainty_cis_ranking.csv", index=False)
    subcomponent_winner_cis(h).round(4).to_csv(OUT / "uncertainty_cis_subcomponents.csv", index=False)
    rerun_sd_cis().round(4).to_csv(OUT / "uncertainty_cis_rerun_sd.csv", index=False)
    print(f"\nSaved 4 CSVs to {OUT}/uncertainty_cis_*.csv")


if __name__ == "__main__":
    main()
