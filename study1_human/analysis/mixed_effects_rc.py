"""Mixed-effects check for the human sub-component regression (reviewer-requested robustness check).

The paper reports pooled OLS: standardized RC ~ 11 sub-components over all 115
human ratings. A reviewer notes ratings are nested within stories and raters and
suggests a mixed-effects model. Each rater rated exactly one story, so a rater
random effect is not estimable (one observation per rater); the estimable
nesting is ratings-within-stories. We refit with a random intercept per story
and compare the fixed-effect weights to the pooled OLS betas.

Run: python3 mixed_effects_rc.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent.parent

SUBS = ["R_Emotion", "A_Topic", "N_Vocab", "N_Plot", "N_Surprise", "R_Empathy",
        "R_Thought", "V_Engagement", "V_Style", "V_Logic", "A_Tone"]


def main():
    df = pd.read_csv(BASE / "data/human_ratings.csv")
    df["story_id"] = df["TOPIC"].str.strip() + "_" + df["TONE"].str.strip()
    d = df[["story_id", "O_Final_Creativity"] + SUBS].dropna().copy()
    print(f"n = {len(d)} ratings, {d.story_id.nunique()} stories")

    # standardize X and y, matching phase1c_subcomponent_weights.py
    for c in SUBS + ["O_Final_Creativity"]:
        d[c] = (d[c] - d[c].mean()) / d[c].std()
    d = d.rename(columns={"O_Final_Creativity": "RC"})

    # pooled OLS (paper's model)
    ols = sm.OLS(d["RC"], sm.add_constant(d[SUBS])).fit()

    # mixed model: same fixed effects, random intercept per story
    mlm = smf.mixedlm("RC ~ " + " + ".join(SUBS), d, groups=d["story_id"]).fit(reml=True)
    story_var = mlm.cov_re.iloc[0, 0]
    icc = story_var / (story_var + mlm.scale)

    print(f"\npooled OLS:   R2 = {ols.rsquared:.3f} (adj {ols.rsquared_adj:.3f})")
    print(f"mixed model:  story-intercept var = {story_var:.4f}, residual var = {mlm.scale:.4f}, ICC = {icc:.3f}")

    comp = pd.DataFrame({
        "beta_OLS": ols.params[SUBS],
        "SE_OLS": ols.bse[SUBS],
        "beta_Mixed": mlm.params[SUBS],
        "SE_Mixed": mlm.bse[SUBS],
    })
    comp["abs_diff"] = (comp.beta_OLS - comp.beta_Mixed).abs()
    comp = comp.sort_values("beta_OLS", ascending=False).round(3)
    print("\n" + comp.to_string())

    b1, b2 = ols.params[SUBS].values, mlm.params[SUBS].values
    cos = b1 @ b2 / (np.linalg.norm(b1) * np.linalg.norm(b2))
    rho, _ = spearmanr(b1, b2)
    print(f"\ncosine(beta_OLS, beta_Mixed) = {cos:.4f}")
    print(f"Spearman rank rho            = {rho:.4f}")
    print(f"max |difference|             = {comp.abs_diff.max():.3f}")

    out = Path(__file__).parent / "mixed_effects_rc_comparison.csv"
    comp.to_csv(out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
