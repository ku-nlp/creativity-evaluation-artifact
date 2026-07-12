"""
Phase E: Mixed-Effects Model

E1: IC ~ tone + (1|story) + (1|family)
    gk_flip ~ tone + (1|story) + (1|family)
E2: Variance decomposition (ICC for random effects)

Output: console summary
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def load_data():
    llm = pd.read_csv(BASE / "data/llm_ratings.csv")
    # Ensure tone is categorical
    llm["tone"] = pd.Categorical(llm["tone"], categories=["clinical", "melancholic", "surreal", "witty"])
    return llm


def fit_mixed_model(llm, formula, title):
    """Fit a mixed-effects model and print results."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    model = smf.mixedlm(formula, llm, groups=llm["family"], re_formula="1",
                         vc_formula={"story_id": "0 + C(story_id)"})
    result = model.fit(reml=True)
    print(result.summary())

    return result


def variance_decomposition(llm, outcome):
    """
    Compute ICC for family and story random effects.
    Uses one-way random effects ANOVA approach.
    """
    print(f"\n  --- Variance Decomposition for {outcome} ---")

    # Family-level ICC
    grand_mean = llm[outcome].mean()
    family_means = llm.groupby("family")[outcome].mean()
    story_means = llm.groupby("story_id")[outcome].mean()

    n_families = llm["family"].nunique()
    n_stories = llm["story_id"].nunique()
    n_total = len(llm)

    # Between-family variance
    n_per_family = llm.groupby("family").size()
    ss_family = sum(n_per_family[f] * (family_means[f] - grand_mean)**2 for f in family_means.index)
    ms_family = ss_family / (n_families - 1)

    # Between-story variance
    n_per_story = llm.groupby("story_id").size()
    ss_story = sum(n_per_story[s] * (story_means[s] - grand_mean)**2 for s in story_means.index)
    ms_story = ss_story / (n_stories - 1)

    # Residual variance
    ss_total = ((llm[outcome] - grand_mean)**2).sum()
    ss_residual = ss_total - ss_family - ss_story
    df_residual = n_total - n_families - n_stories + 1
    ms_residual = ss_residual / df_residual if df_residual > 0 else 0

    total_var = ms_family + ms_story + ms_residual
    if total_var > 0:
        icc_family = ms_family / total_var
        icc_story = ms_story / total_var
        icc_residual = ms_residual / total_var
    else:
        icc_family = icc_story = icc_residual = 0

    print(f"  {'Source':<15} {'MS':>10} {'% Variance':>12}")
    print(f"  {'-'*40}")
    print(f"  {'Family':<15} {ms_family:>10.3f} {icc_family:>11.1%}")
    print(f"  {'Story':<15} {ms_story:>10.3f} {icc_story:>11.1%}")
    print(f"  {'Residual':<15} {ms_residual:>10.3f} {icc_residual:>11.1%}")
    print(f"  {'Total':<15} {total_var:>10.3f} {'100.0%':>12}")

    return {"family": icc_family, "story": icc_story, "residual": icc_residual}


def main():
    llm = load_data()
    print(f"LLM data: {len(llm)} rows, {llm['model_label'].nunique()} conditions")
    print(f"Families: {llm['family'].unique()}")
    print(f"Stories: {llm['story_id'].nunique()}")

    # ── E1: Mixed-effects models ─────────────────────────────────────────
    # Model 1: What predicts initial creativity severity?
    fit_mixed_model(
        llm,
        "IC ~ C(tone, Treatment(reference='clinical'))",
        "Model 1: IC ~ tone + (1|family) + (1|story)"
    )

    # Model 2: What predicts the gatekeeper flip?
    fit_mixed_model(
        llm,
        "gk_flip ~ C(tone, Treatment(reference='clinical'))",
        "Model 2: gk_flip ~ tone + (1|family) + (1|story)"
    )

    # ── E2: Variance decomposition ──────────────────────────────────────
    print("\n" + "="*60)
    print("  E2: Variance Decomposition")
    print("="*60)

    for outcome in ["IC", "RC", "gk_flip", "enjoyment"]:
        variance_decomposition(llm, outcome)

    # ── Summary: what matters most? ──────────────────────────────────────
    print("\n" + "="*60)
    print("  Summary: What Drives Variation?")
    print("="*60)

    for outcome in ["IC", "gk_flip"]:
        decomp = variance_decomposition(llm, outcome)
        dominant = max(decomp, key=decomp.get)
        print(f"\n  {outcome}: dominant source = {dominant} ({decomp[dominant]:.1%})")
        if dominant == "residual":
            print(f"    → Most variation is within-family, within-story (model-specific)")
        elif dominant == "family":
            print(f"    → Model family is the main driver")
        elif dominant == "story":
            print(f"    → Story content is the main driver")


if __name__ == "__main__":
    main()
