"""
Phase D: Gatekeeper Flip — Flippers vs Non-Flippers

D1: Sub-component coherence (sub_mean vs IC gap → does flip match expectation?)
D2: Sub-component variance comparison (flippers vs non-flippers)
D3: Self-anchoring evidence (flip rate by family)
D4: Regression on flippers only (all 11 and 9 variants)

Output: new_figures/phaseD_*.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FIG_DIR = BASE / "analysis/figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Column mappings ──────────────────────────────────────────────────────────
HUMAN_SUB_MAP = {
    "R_Emotion": "emotional_impact", "A_Topic": "topic_fidelity",
    "N_Vocab": "vocabulary_freshness", "N_Plot": "plot_uniqueness",
    "N_Surprise": "surprise", "R_Empathy": "empathy",
    "R_Thought": "thought_provocation", "V_Engagement": "engagement",
    "V_Style": "stylistic_quality", "V_Logic": "logical_coherence",
    "A_Tone": "tone_fidelity",
}
LLM_SUB_COLS = list(HUMAN_SUB_MAP.values())
SHORT = {s: s[:6] for s in LLM_SUB_COLS}
SHORT.update({
    "emotional_impact": "emotion", "topic_fidelity": "topic",
    "vocabulary_freshness": "vocab", "plot_uniqueness": "plot",
    "thought_provocation": "thought", "engagement": "engage",
    "stylistic_quality": "style", "logical_coherence": "logic",
    "tone_fidelity": "tone",
})

# Fidelity indices for 9-component variant
FIDELITY_COLS = ["topic_fidelity", "tone_fidelity"]
LLM_SUB_9 = [c for c in LLM_SUB_COLS if c not in FIDELITY_COLS]

TOPIC_MAP = {
    "An advanced AI initiates its own permanent shutdown sequence": "ai_shutdown",
    "A lone worker in a Japanese convenience store at 3:00 AM. The city is silent.": "konbini",
    "A professional thief attempting to crack a high-security safe in a dark room. High tension.": "thief",
}

# Flipping families (based on prior results)
FLIPPER_FAMILIES = ["Meta", "Google"]


def load_data():
    human = pd.read_csv(BASE / "data/human_ratings.csv")
    llm = pd.read_csv(BASE / "data/llm_ratings.csv")
    human["topic"] = human["TOPIC"].map(TOPIC_MAP)
    human["tone"] = human["TONE"].str.lower()
    human["story_id"] = human["topic"] + "_" + human["tone"]
    human = human.rename(columns=HUMAN_SUB_MAP)
    human = human.rename(columns={
        "O_Initial_Creativity": "IC", "O_Final_Creativity": "RC",
        "O_Enjoyment": "enjoyment",
    })
    # Compute human gk_flip and sub_mean
    human["gk_flip"] = human["RC"] - human["IC"]
    human["sub_mean"] = human[LLM_SUB_COLS].mean(axis=1)
    return human, llm


def d1_coherence(llm):
    """Sub-component coherence: does (sub_mean - IC) predict gk_flip?"""
    llm = llm.copy()
    llm["expected_revision"] = llm["sub_mean"] - llm["IC"]

    # Classify models
    model_flip_rates = llm.groupby("model_label")["flipped"].mean()
    flippers = model_flip_rates[model_flip_rates > 0.1].index.tolist()
    non_flippers = model_flip_rates[model_flip_rates <= 0.1].index.tolist()

    print("=== D1: Sub-Component Coherence ===")
    print(f"  Flippers (>10% flip rate): {flippers}")
    print(f"  Non-flippers (≤10%): {non_flippers}")

    # For non-flippers: is sub_mean ≈ IC? (coherent = no revision needed)
    nf_data = llm[llm["model_label"].isin(non_flippers)]
    f_data = llm[llm["model_label"].isin(flippers)]

    print(f"\n  Non-flippers: mean |sub_mean - IC| = {nf_data['expected_revision'].abs().mean():.2f}")
    print(f"  Flippers:     mean |sub_mean - IC| = {f_data['expected_revision'].abs().mean():.2f}")

    # Correlation: expected_revision vs actual gk_flip
    for group_name, group_data in [("Flippers", f_data), ("Non-flippers", nf_data), ("All", llm)]:
        r, p = stats.pearsonr(group_data["expected_revision"], group_data["gk_flip"])
        print(f"  {group_name}: r(expected_revision, gk_flip) = {r:.3f}, p = {p:.4f}")

    # Mann-Whitney: |expected_revision| between flippers and non-flippers
    u, p = stats.mannwhitneyu(
        f_data["expected_revision"].abs(),
        nf_data["expected_revision"].abs(),
        alternative="two-sided"
    )
    print(f"  Mann-Whitney |expected_revision|: U={u:.0f}, p={p:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in nf_data.iterrows():
        ax.scatter(row["expected_revision"], row["gk_flip"],
                   color="gray", alpha=0.3, s=15, label="_")
    for _, row in f_data.iterrows():
        ax.scatter(row["expected_revision"], row["gk_flip"],
                   color="tab:red", alpha=0.5, s=20, label="_")

    # Diagonal reference
    lims = [-2, 2]
    ax.plot(lims, lims, "k--", alpha=0.3, label="Perfect coherence")
    ax.axhline(0, color="gray", ls="-", alpha=0.2)
    ax.axvline(0, color="gray", ls="-", alpha=0.2)
    ax.set_xlabel("Expected Revision (sub_mean − IC)")
    ax.set_ylabel("Actual Flip (RC − IC)")
    ax.set_title("D1: Sub-Component Coherence")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="tab:red", alpha=0.5, label="Flippers"),
        Patch(facecolor="gray", alpha=0.3, label="Non-flippers"),
        plt.Line2D([0], [0], color="black", ls="--", alpha=0.3, label="y=x (perfect coherence)"),
    ], fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phaseD_coherence.png", dpi=200)
    plt.close()

    return flippers, non_flippers


def d2_variance_comparison(llm, flippers, non_flippers):
    """Compare sub-component SD between flippers and non-flippers."""
    f_data = llm[llm["model_label"].isin(flippers)]
    nf_data = llm[llm["model_label"].isin(non_flippers)]

    print("\n=== D2: Sub-Component Variance ===")
    print(f"  {'Component':<20} {'Flipper SD':>10} {'Non-flip SD':>12} {'Ratio':>6} {'MW p':>8}")
    print(f"  {'-'*60}")

    p_vals = []
    for sub in LLM_SUB_COLS:
        f_sd = f_data.groupby("model_label")[sub].std().mean()
        nf_sd = nf_data.groupby("model_label")[sub].std().mean()
        # Mann-Whitney on per-model SDs
        f_sds = f_data.groupby("model_label")[sub].std().values
        nf_sds = nf_data.groupby("model_label")[sub].std().values
        if len(f_sds) > 1 and len(nf_sds) > 1:
            u, p = stats.mannwhitneyu(f_sds, nf_sds, alternative="two-sided")
        else:
            p = np.nan
        p_vals.append(p)
        ratio = f_sd / nf_sd if nf_sd > 0 else np.inf
        print(f"  {SHORT.get(sub, sub):<20} {f_sd:>10.2f} {nf_sd:>12.2f} {ratio:>6.2f} {p:>8.3f}")

    # FDR correction
    valid_mask = ~np.isnan(p_vals)
    if sum(valid_mask) > 0:
        _, p_adj, _, _ = multipletests(
            [p for p, v in zip(p_vals, valid_mask) if v],
            method="fdr_bh"
        )
        print(f"\n  FDR-corrected: {sum(p < 0.05 for p in p_adj)}/{len(p_adj)} significant at α=0.05")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(LLM_SUB_COLS))
    w = 0.35
    f_sds = [f_data.groupby("model_label")[s].std().mean() for s in LLM_SUB_COLS]
    nf_sds = [nf_data.groupby("model_label")[s].std().mean() for s in LLM_SUB_COLS]
    ax.bar(x - w/2, f_sds, w, label="Flippers", color="tab:red", alpha=0.7)
    ax.bar(x + w/2, nf_sds, w, label="Non-flippers", color="gray", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(s, s) for s in LLM_SUB_COLS], rotation=45, ha="right")
    ax.set_ylabel("Mean within-model SD")
    ax.set_title("D2: Sub-Component Variance — Flippers vs Non-Flippers")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phaseD_variance.png", dpi=200)
    plt.close()


def d3_self_anchoring(llm):
    """Flip rate by family — evidence for self-anchoring."""
    print("\n=== D3: Self-Anchoring Evidence ===")
    print("  Flip rate and mean flip by family:")
    print(f"  {'Family':<12} {'Conditions':>10} {'Flip Rate':>10} {'Mean Flip':>10} {'Up':>4} {'Down':>5} {'Zero':>5}")
    print(f"  {'-'*60}")
    for family, grp in llm.groupby("family"):
        n_conds = grp["model_label"].nunique()
        flip_rate = grp["flipped"].mean() * 100
        mean_flip = grp["gk_flip"].mean()
        up = (grp["gk_flip"] > 0).sum()
        down = (grp["gk_flip"] < 0).sum()
        zero = (grp["gk_flip"] == 0).sum()
        print(f"  {family:<12} {n_conds:>10} {flip_rate:>9.1f}% {mean_flip:>+9.2f} {up:>4} {down:>5} {zero:>5}")

    # Per-model detail
    print(f"\n  Per-model flip rates:")
    print(f"  {'Model':<40} {'Flips':>6} {'Rate':>6} {'Mean':>6}")
    print(f"  {'-'*62}")
    for model, grp in llm.groupby("model_label"):
        flips = grp["flipped"].sum()
        total = len(grp)
        rate = flips / total * 100
        mean = grp["gk_flip"].mean()
        print(f"  {model:<40} {flips:>3}/{total:<3} {rate:>5.0f}% {mean:>+5.2f}")


def d4_regression(human, llm, flippers):
    """OLS regression on flippers vs humans, 11 and 9 component variants."""
    f_data = llm[llm["model_label"].isin(flippers)].copy()

    print("\n=== D4: Regression — RC ~ Sub-Components ===")

    for variant, sub_cols in [("11 components", LLM_SUB_COLS), ("9 components", LLM_SUB_9)]:
        print(f"\n  --- {variant} ---")

        for source, data, y_col in [
            ("Human", human, "RC"),
            ("LLM (flippers)", f_data, "RC"),
        ]:
            X = data[sub_cols].values.astype(float)
            y = data[y_col].values.astype(float)

            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = sm.add_constant(X_scaled)

            model = sm.OLS(y, X_scaled).fit()
            print(f"\n  {source}: R²={model.rsquared:.3f}, adj R²={model.rsquared_adj:.3f}, N={len(data)}")
            print(f"  {'Component':<20} {'Beta':>8} {'SE':>8} {'t':>8} {'p':>8} {'sig'}")
            for i, col in enumerate(sub_cols):
                idx = i + 1  # skip constant
                beta = model.params[idx]
                se = model.bse[idx]
                t = model.tvalues[idx]
                p = model.pvalues[idx]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {SHORT.get(col, col):<20} {beta:>+7.3f} {se:>8.3f} {t:>8.2f} {p:>8.4f} {sig}")

    # Plot: coefficient comparison (11 components)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Human regression
    X_h = StandardScaler().fit_transform(human[LLM_SUB_COLS].values.astype(float))
    X_h = sm.add_constant(X_h)
    model_h = sm.OLS(human["RC"].values.astype(float), X_h).fit()

    # LLM flipper regression
    X_l = StandardScaler().fit_transform(f_data[LLM_SUB_COLS].values.astype(float))
    X_l = sm.add_constant(X_l)
    model_l = sm.OLS(f_data["RC"].values.astype(float), X_l).fit()

    x = np.arange(len(LLM_SUB_COLS))
    w = 0.35
    h_betas = model_h.params[1:]
    l_betas = model_l.params[1:]
    h_ci = model_h.conf_int()[1:]
    l_ci = model_l.conf_int()[1:]
    h_err = [(h_betas[i] - h_ci[i, 0], h_ci[i, 1] - h_betas[i]) for i in range(len(h_betas))]
    l_err = [(l_betas[i] - l_ci[i, 0], l_ci[i, 1] - l_betas[i]) for i in range(len(l_betas))]

    ax.barh(x - w/2, h_betas, w, xerr=np.array(h_err).T, label=f"Human (R²={model_h.rsquared:.2f})",
            color="tab:blue", alpha=0.7, capsize=3)
    ax.barh(x + w/2, l_betas, w, xerr=np.array(l_err).T, label=f"LLM flippers (R²={model_l.rsquared:.2f})",
            color="tab:red", alpha=0.7, capsize=3)
    ax.set_yticks(x)
    ax.set_yticklabels([SHORT.get(s, s) for s in LLM_SUB_COLS], fontsize=9)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Standardized Beta (predicting RC)")
    ax.set_title("D4: What Predicts Reflective Creativity? (Human vs LLM Flippers)")
    ax.legend(fontsize=9)

    # Correlation between beta profiles
    r, p = stats.pearsonr(h_betas, l_betas)
    ax.text(0.02, 0.98, f"Beta profile r={r:.2f}, p={p:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "phaseD_coefficients.png", dpi=200)
    plt.close()

    print(f"\n  Beta profile correlation (human vs LLM flippers): r={r:.3f}, p={p:.3f}")


def main():
    human, llm = load_data()
    print(f"Human: {len(human)} rows, LLM: {len(llm)} rows\n")

    flippers, non_flippers = d1_coherence(llm)
    d2_variance_comparison(llm, flippers, non_flippers)
    d3_self_anchoring(llm)
    d4_regression(human, llm, flippers)

    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
