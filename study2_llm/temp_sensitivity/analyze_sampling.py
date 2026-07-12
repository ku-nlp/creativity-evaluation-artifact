"""Analyze repeated-sampling / temperature-sensitivity runs (reviewer-requested sampling robustness check).

Reads every temp_sens_*.json and sampling_*.json in results/ and reports,
per (condition, temperature):
  - mean within-story SD of RC and IC across runs (the sampling noise)
  - share of stories where every run returned the identical RC
  - mean RC (to check for drift across temperatures)
Reference: the same within-story SD for human raters (between-rater
disagreement on the shared 12 stories), computed live from human_ratings.csv.

Run: python3 study2_llm/temp_sensitivity/analyze_sampling.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
HUMAN_CSV = HERE.parent.parent / "study1_human" / "data" / "human_ratings.csv"
OUT = HERE.parent / "analysis"


def load(path):
    d = json.loads(path.read_text())
    rows = []
    for r in d.get("results", []):
        if not r.get("parse_ok"):
            continue
        rows.append({"story_id": r["story_id"], "run_idx": r["run_idx"],
                     "temperature": r.get("temperature"),
                     "IC": r["scores"]["initial_creativity"],
                     "RC": r["scores"]["reflective_creativity"]})
    df = pd.DataFrame(rows)
    df["condition"] = d.get("label", path.stem)
    df["model"] = d.get("model", "?")
    return df


def summarize(df):
    """Per-story stats across runs, then averaged over stories."""
    g = df.groupby("story_id")
    per_story = g.agg(sd_RC=("RC", "std"), sd_IC=("IC", "std"),
                      n_runs=("RC", "size"),
                      identical=("RC", lambda x: x.nunique() == 1))
    multi = per_story[per_story.n_runs >= 2]
    return {"n_stories": len(multi),
            "runs_per_story": multi.n_runs.mean(),
            "sd_RC": multi.sd_RC.mean(),
            "sd_IC": multi.sd_IC.mean(),
            "pct_identical_RC": 100 * multi.identical.mean(),
            "mean_RC": df.RC.mean()}


def main():
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        df = load(path)
        if df.empty:
            continue
        temp = df.temperature.iloc[0]
        rows.append({"file": path.name, "condition": df.condition.iloc[0],
                     "model": df.model.iloc[0],
                     "temperature": "default" if temp is None or pd.isna(temp)
                                    else f"{temp:g}",
                     **summarize(df)})
    t = pd.DataFrame(rows).set_index("file")

    # human reference: between-rater SD within story, mean over the 12 stories
    h = pd.read_csv(HUMAN_CSV)
    hs = h.groupby(["TOPIC", "TONE"]).agg(
        sd_RC=("O_Final_Creativity", "std"), sd_IC=("O_Initial_Creativity", "std"))
    print(f"HUMAN reference (between-rater, within-story, 12 stories): "
          f"SD(RC) = {hs.sd_RC.mean():.2f}, SD(IC) = {hs.sd_IC.mean():.2f}\n")

    pd.set_option("display.width", 200)
    print(t.round(3).to_string())

    OUT.mkdir(exist_ok=True)
    t.round(4).to_csv(OUT / "sampling_stability_summary.csv")
    print(f"\nSaved: {OUT / 'sampling_stability_summary.csv'}")


if __name__ == "__main__":
    main()
