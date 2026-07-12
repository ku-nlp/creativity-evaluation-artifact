"""
Phase 1C: Sub-component weighting regression.

What predicts final creativity (RC) for humans vs LLMs?
OLS: RC ~ 11 sub-components (standardized betas).

Produces:
  1. Coefficient comparison bar chart (human vs LLM betas)
  2. Per-family coefficient heatmap
  3. Full regression tables printed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# --- Column mapping ---
HUMAN_TO_LLM = {
    "O_Final_Creativity": "RC",
    "O_Initial_Creativity": "IC",
    "R_Emotion": "emotional_impact",
    "A_Topic": "topic_fidelity",
    "N_Vocab": "vocabulary_freshness",
    "N_Plot": "plot_uniqueness",
    "N_Surprise": "surprise",
    "R_Empathy": "empathy",
    "R_Thought": "thought_provocation",
    "V_Engagement": "engagement",
    "V_Style": "stylistic_quality",
    "V_Logic": "logical_coherence",
    "A_Tone": "tone_fidelity",
}

SUB_COMPONENTS = [
    "emotional_impact", "topic_fidelity", "vocabulary_freshness",
    "plot_uniqueness", "surprise", "empathy", "thought_provocation",
    "engagement", "stylistic_quality", "logical_coherence", "tone_fidelity",
]

SHORT_LABELS = {
    "emotional_impact": "Emotion",
    "topic_fidelity": "Topic",
    "vocabulary_freshness": "Vocab",
    "plot_uniqueness": "Plot",
    "surprise": "Surprise",
    "empathy": "Empathy",
    "thought_provocation": "Thought",
    "engagement": "Engage",
    "stylistic_quality": "Style",
    "logical_coherence": "Logic",
    "tone_fidelity": "Tone",
}

# --- Load data ---
base = Path(__file__).resolve().parent.parent
human_raw = pd.read_csv(base / "data/human_ratings.csv").rename(columns=HUMAN_TO_LLM)
llm = pd.read_csv(base / "data/llm_ratings.csv")


def run_ols(df, y_col, x_cols, label=""):
    """Run OLS with standardized predictors, return betas and stats."""
    df_clean = df[[y_col] + x_cols].dropna()
    if len(df_clean) < len(x_cols) + 2:
        print(f"  WARNING: {label} has only {len(df_clean)} rows for {len(x_cols)} predictors")
        return None

    X = df_clean[x_cols].values
    y = df_clean[y_col].values

    # Drop zero-variance columns (e.g., tone_fidelity always 7 for some families)
    col_stds = df_clean[x_cols].std()
    kept_cols = [c for c in x_cols if col_stds[c] > 0]
    dropped = [c for c in x_cols if col_stds[c] == 0]
    if dropped:
        print(f"  Dropped zero-variance: {dropped}")

    X = df_clean[kept_cols].values
    y = df_clean[y_col].values

    # Standardize X (z-scores) so betas are comparable
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X)

    # Also standardize y for fully standardized betas
    y_mean, y_std = y.mean(), y.std()
    y_z = (y - y_mean) / y_std if y_std > 0 else y - y_mean

    # OLS
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X_z, y_z)

    # Predictions and R²
    y_pred = reg.predict(X_z)
    ss_res = np.sum((y_z - y_pred) ** 2)
    ss_tot = np.sum((y_z - y_z.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    n = len(y_z)
    k = len(kept_cols)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    # Standard errors for betas (standardized)
    mse = ss_res / (n - k - 1) if n - k - 1 > 0 else 1e-10
    XtX_inv = np.linalg.pinv(X_z.T @ X_z)
    se_betas = np.sqrt(np.maximum(mse * np.diag(XtX_inv), 1e-10))
    t_stats = reg.coef_ / se_betas
    p_values = 2 * stats.t.sf(np.abs(t_stats), df=max(n - k - 1, 1))

    # Build results with all original predictors (zero beta for dropped ones)
    rows_out = []
    j = 0
    for c in x_cols:
        if c in kept_cols:
            rows_out.append({"predictor": c, "beta": reg.coef_[j],
                             "se": se_betas[j], "t": t_stats[j], "p": p_values[j]})
            j += 1
        else:
            rows_out.append({"predictor": c, "beta": 0.0, "se": 0.0, "t": 0.0, "p": 1.0})

    results = pd.DataFrame(rows_out)
    results["sig"] = results["p"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )

    return {"results": results, "r2": r2, "r2_adj": r2_adj, "n": n, "label": label}


# =====================================================================
# 1. Human regression: RC ~ 11 sub-components (N=115)
# =====================================================================
human_ols = run_ols(human_raw, "RC", SUB_COMPONENTS, label="Human (N=115)")

# =====================================================================
# 2. LLM pooled regression: RC ~ 11 sub-components (N=204)
# =====================================================================
llm_ols = run_ols(llm, "RC", SUB_COMPONENTS, label="LLM pooled (N=204)")

# =====================================================================
# 3. Per-family LLM regressions
# =====================================================================
family_ols = {}
for family in sorted(llm["family"].unique()):
    fam_df = llm[llm["family"] == family]
    n = len(fam_df)
    result = run_ols(fam_df, "RC", SUB_COMPONENTS, label=f"{family} (N={n})")
    if result is not None:
        family_ols[family] = result

# =====================================================================
# Print results
# =====================================================================
print("=" * 70)
print("PHASE 1C: SUB-COMPONENT WEIGHTING REGRESSION")
print("=" * 70)

for ols in [human_ols, llm_ols] + list(family_ols.values()):
    if ols is None:
        continue
    print(f"\n--- {ols['label']} ---")
    print(f"  R² = {ols['r2']:.3f}, Adj R² = {ols['r2_adj']:.3f}, N = {ols['n']}")
    print(f"  {'Predictor':25s} {'Beta':>7s} {'SE':>7s} {'t':>7s} {'p':>8s}")
    for _, row in ols["results"].sort_values("beta", ascending=False).iterrows():
        print(f"  {row['predictor']:25s} {row['beta']:+7.3f} {row['se']:7.3f} "
              f"{row['t']:+7.2f} {row['p']:8.4f} {row['sig']}")

# =====================================================================
# 4. Figure: Coefficient comparison — Human vs LLM (Fig 5 candidate)
# =====================================================================
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(SUB_COMPONENTS))
width = 0.35

human_betas = human_ols["results"].set_index("predictor").reindex(SUB_COMPONENTS)["beta"]
llm_betas = llm_ols["results"].set_index("predictor").reindex(SUB_COMPONENTS)["beta"]
human_sigs = human_ols["results"].set_index("predictor").reindex(SUB_COMPONENTS)["sig"]
llm_sigs = llm_ols["results"].set_index("predictor").reindex(SUB_COMPONENTS)["sig"]

bars1 = ax.bar(x - width / 2, human_betas, width, label="Human", color="#e67e22",
               edgecolor="white", alpha=0.85)
bars2 = ax.bar(x + width / 2, llm_betas, width, label="LLM (pooled)", color="#2980b9",
               edgecolor="white", alpha=0.85)

# Significance markers
for i, sc in enumerate(SUB_COMPONENTS):
    if human_sigs[sc]:
        ax.text(i - width / 2, human_betas[sc] + 0.02 * np.sign(human_betas[sc]),
                human_sigs[sc], ha="center", va="bottom" if human_betas[sc] >= 0 else "top",
                fontsize=8, fontweight="bold", color="#c0392b")
    if llm_sigs[sc]:
        ax.text(i + width / 2, llm_betas[sc] + 0.02 * np.sign(llm_betas[sc]),
                llm_sigs[sc], ha="center", va="bottom" if llm_betas[sc] >= 0 else "top",
                fontsize=8, fontweight="bold", color="#2471a3")

ax.set_xticks(x)
ax.set_xticklabels([SHORT_LABELS[sc] for sc in SUB_COMPONENTS], fontsize=10, rotation=30, ha="right")
ax.set_ylabel("Standardized β", fontsize=12)
ax.axhline(0, color="black", linewidth=0.5)
ax.legend(fontsize=11)
ax.grid(True, axis="y", alpha=0.2)

ax.set_title(
    f"What Predicts Final Creativity?\n"
    f"Human R²={human_ols['r2']:.2f}  |  LLM R²={llm_ols['r2']:.2f}",
    fontsize=14, fontweight="bold"
)

fig.tight_layout()
fig.savefig(OUT / "1c_coefficient_comparison.png", dpi=200, bbox_inches="tight")
print(f"\nSaved: {OUT / '1c_coefficient_comparison.png'}")

# =====================================================================
# 5. Figure: Per-family coefficient heatmap
# =====================================================================
all_labels = ["Human"] + sorted(family_ols.keys())
all_results = [human_ols] + [family_ols[f] for f in sorted(family_ols.keys())]

beta_matrix = np.zeros((len(all_labels), len(SUB_COMPONENTS)))
for i, ols in enumerate(all_results):
    if ols is None:
        continue
    betas = ols["results"].set_index("predictor").reindex(SUB_COMPONENTS)["beta"]
    beta_matrix[i, :] = betas.values

fig, ax = plt.subplots(figsize=(14, 5))
cmap = plt.cm.RdBu_r
im = ax.imshow(beta_matrix, cmap=cmap, aspect="auto", vmin=-0.6, vmax=0.6)

for i in range(beta_matrix.shape[0]):
    for j in range(beta_matrix.shape[1]):
        val = beta_matrix[i, j]
        # Get significance
        sig = all_results[i]["results"].set_index("predictor").loc[SUB_COMPONENTS[j], "sig"]
        text_color = "white" if abs(val) > 0.4 else "black"
        ax.text(j, i, f"{val:+.2f}{sig}", ha="center", va="center",
                fontsize=8, color=text_color)

ax.set_xticks(range(len(SUB_COMPONENTS)))
ax.set_xticklabels([SHORT_LABELS[sc] for sc in SUB_COMPONENTS], fontsize=9, rotation=30, ha="right")
ax.set_yticks(range(len(all_labels)))
r2_labels = [f"{lbl} (R²={r['r2']:.2f})" for lbl, r in zip(all_labels, all_results)]
ax.set_yticklabels(r2_labels, fontsize=10)

ax.axhline(0.5, color="black", linewidth=2)  # Separator after human

cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Standardized β")
ax.set_title("Sub-Component Weights: Human vs LLM Families", fontsize=14, fontweight="bold")

fig.tight_layout()
fig.savefig(OUT / "1c_family_heatmap.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT / '1c_family_heatmap.png'}")

# =====================================================================
# 6. Key comparisons
# =====================================================================
print("\n--- Key Beta Comparisons (Human vs LLM pooled) ---")
merged_betas = pd.DataFrame({
    "human_beta": human_betas,
    "llm_beta": llm_betas,
    "delta": llm_betas - human_betas,
}).sort_values("delta", ascending=False)

for sc, row in merged_betas.iterrows():
    direction = "LLM weights MORE" if row["delta"] > 0 else "Human weights MORE"
    print(f"  {SHORT_LABELS[sc]:10s}  human={row['human_beta']:+.3f}  "
          f"llm={row['llm_beta']:+.3f}  Δ={row['delta']:+.3f}  ({direction})")

# Correlation between human and LLM beta profiles
r_betas = np.corrcoef(human_betas, llm_betas)[0, 1]
rho_betas, p_rho = stats.spearmanr(human_betas, llm_betas)
print(f"\n  Beta profile correlation: r = {r_betas:+.3f}")
print(f"  Beta profile Spearman:   ρ = {rho_betas:+.3f}, p = {p_rho:.4f}")

print("\n" + "=" * 70)
print("Phase 1C complete. Figures saved to study1_human/analysis/figures/")
print("=" * 70)
