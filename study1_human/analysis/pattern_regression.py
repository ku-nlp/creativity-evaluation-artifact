"""
Pattern Analysis on Human Data (N=115 raters, 12 stories)
Two-stage approach:
  Stage 1 — Regression: which sub-components predict RC (rater-level + story-level)
  Stage 2 — Clustering: rater profiles by sub-component pattern, check if clusters predict RC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Setup ──────────────────────────────────────────────────────────────────────
OUT = "study1_human/analysis/figures/"

df = pd.read_csv("study1_human/data/human_ratings.csv")

# Short topic labels
TOPIC_MAP = {
    "An advanced AI initiates its own permanent shutdown sequence": "ai_shutdown",
    "A lone worker in a Japanese convenience store at 3:00 AM. The city is silent.": "konbini",
    "A professional thief attempting to crack a high-security safe in a dark room. High tension.": "thief",
}
df["topic"] = df["TOPIC"].map(TOPIC_MAP)
df["story_id"] = df["topic"] + "_" + df["TONE"].str.lower()

SUBS = ["R_Emotion", "A_Topic", "N_Vocab", "N_Plot", "N_Surprise",
        "R_Empathy", "R_Thought", "V_Engagement", "V_Style", "V_Logic", "A_Tone"]

SUB_LABELS = {
    "R_Emotion": "Emotion", "A_Topic": "Topic\nFidelity", "N_Vocab": "Vocab",
    "N_Plot": "Plot", "N_Surprise": "Surprise", "R_Empathy": "Empathy",
    "R_Thought": "Thought\nProvocation", "V_Engagement": "Engagement",
    "V_Style": "Style", "V_Logic": "Logic", "A_Tone": "Tone\nFidelity"
}

IC_COL = "O_Initial_Creativity"
RC_COL = "O_Final_Creativity"

# ICC weights from Phase B (empirical from human data)
# Positive → reliable signal; negative/near-zero → noisy
ICC_WEIGHTS_RAW = {
    "R_Emotion": 0.66, "A_Topic": 0.30, "N_Vocab": -0.26, "N_Plot": 0.42,
    "N_Surprise": 0.69, "R_Empathy": 0.69, "R_Thought": 0.40,
    "V_Engagement": 0.55, "V_Style": 0.35, "V_Logic": 0.28, "A_Tone": -1.02,
}
# Clip negatives to 0, normalize to sum to 1
ICC_W = np.array([max(0, ICC_WEIGHTS_RAW[s]) for s in SUBS])
ICC_W = ICC_W / ICC_W.sum()

print("=" * 60)
print("PATTERN ANALYSIS — HUMAN DATA")
print("=" * 60)
print(f"N raters: {len(df)}  |  N stories: {df['story_id'].nunique()}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1A — RATER-LEVEL REGRESSION: RC ~ 11 sub-components
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Stage 1A: Rater-level OLS  RC ~ 11 sub-components ──")

X = sm.add_constant(df[SUBS])
y = df[RC_COL]
model_full = sm.OLS(y, X).fit()

print(model_full.summary2().tables[1].to_string())
print(f"\nR² = {model_full.rsquared:.3f}  |  Adj-R² = {model_full.rsquared_adj:.3f}")

# FDR correction on sub-component p-values
pvals = model_full.pvalues[SUBS]
reject, pvals_fdr, _, _ = multipletests(pvals, method="fdr_bh")
fdr_df = pd.DataFrame({
    "coef": model_full.params[SUBS],
    "p_raw": pvals,
    "p_fdr": pvals_fdr,
    "sig": reject
}, index=SUBS)
print("\nFDR-corrected significance:")
print(fdr_df.sort_values("coef", ascending=False).to_string())

# ── Also run RC ~ IC + 11 sub-components (partial effects beyond IC) ──
print("\n── Stage 1B: RC ~ IC + 11 sub-components (partial effects) ──")
X2 = sm.add_constant(df[[IC_COL] + SUBS])
model_ic = sm.OLS(y, X2).fit()
print(f"R² = {model_ic.rsquared:.3f}  |  Adj-R² = {model_ic.rsquared_adj:.3f}")
print(f"IC coef = {model_ic.params[IC_COL]:.3f}  p = {model_ic.pvalues[IC_COL]:.4f}")

pvals2 = model_ic.pvalues[SUBS]
_, pvals2_fdr, _, _ = multipletests(pvals2, method="fdr_bh")
fdr2_df = pd.DataFrame({
    "partial_coef": model_ic.params[SUBS],
    "p_fdr": pvals2_fdr,
}).sort_values("partial_coef", ascending=False)
print(fdr2_df.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1C — STORY-LEVEL REGRESSION (means)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Stage 1C: Story-level Spearman correlations (N=12) ──")
story_means = df.groupby("story_id")[SUBS + [IC_COL, RC_COL]].mean()

story_corrs = []
for s in SUBS:
    r, p = stats.spearmanr(story_means[s], story_means[RC_COL])
    story_corrs.append({"sub": s, "rho": r, "p": p})
story_corr_df = pd.DataFrame(story_corrs).sort_values("rho", ascending=False)
print(story_corr_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — RATER-LEVEL CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Stage 2: K-Means clustering on sub-component profiles ──")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[SUBS])

# Determine optimal k (k=2..6) via silhouette
sil_scores = {}
inertias = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, labels)
    inertias[k] = km.inertia_

print("Silhouette scores:", {k: f"{v:.3f}" for k, v in sil_scores.items()})
best_k = max(sil_scores, key=sil_scores.get)
print(f"Best k = {best_k} (silhouette = {sil_scores[best_k]:.3f})")

# Fit final clustering
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df["cluster"] = km_final.fit_predict(X_scaled)

# Profile each cluster
cluster_profiles = df.groupby("cluster")[SUBS + [IC_COL, RC_COL]].mean()
cluster_profiles["n_raters"] = df.groupby("cluster").size()
cluster_profiles["IC_RC_diff"] = cluster_profiles[RC_COL] - cluster_profiles[IC_COL]
print("\nCluster profiles (means):")
print(cluster_profiles.round(2).to_string())

# Story composition per cluster
print("\nCluster composition by story:")
comp = pd.crosstab(df["cluster"], df["story_id"])
print(comp.to_string())

# RC difference across clusters (one-way ANOVA)
groups = [df[df["cluster"] == c][RC_COL].values for c in range(best_k)]
f_stat, p_anova = stats.f_oneway(*groups)
print(f"\nANOVA RC across clusters: F={f_stat:.2f}, p={p_anova:.4f}")

# ── IC→RC shift per cluster ──
print("\nIC→RC shift per cluster:")
for c in range(best_k):
    sub = df[df["cluster"] == c]
    shift = (sub[RC_COL] - sub[IC_COL]).mean()
    fliprate = (sub[RC_COL] != sub[IC_COL]).mean()
    print(f"  Cluster {c} (n={len(sub)}): mean shift={shift:+.2f}, flip rate={fliprate:.0%}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2B — ICC-WEIGHTED DISTANCE CLUSTERING (robustness check)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Stage 2B: ICC-weighted clustering (robustness) ──")
X_weighted = X_scaled * (ICC_W * len(SUBS))  # scale weights so mean weight = 1
km_w = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df["cluster_w"] = km_w.fit_predict(X_weighted)

agreement = (df["cluster"] == df["cluster_w"]).mean()
print(f"Cluster label agreement (unweighted vs ICC-weighted): {agreement:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# --- Fig 1: Regression coefficients (rater-level) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Full model coefficients
coefs = fdr_df["coef"]
colors = ["#d62728" if fdr_df.loc[s, "sig"] else "#aec7e8" for s in SUBS]
short_labels = [SUB_LABELS[s] for s in SUBS]
ax = axes[0]
bars = ax.barh(short_labels, coefs.values, color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("OLS Coefficient")
ax.set_title("RC ~ 11 Sub-components\n(red = FDR-significant, rater-level N=115)")
ax.invert_yaxis()

# Partial coefficients (controlling for IC)
partial_coefs = fdr2_df["partial_coef"]
colors2 = ["#d62728" if fdr2_df.loc[s, "p_fdr"] < 0.05 else "#aec7e8"
           for s in fdr2_df.index]
ax2 = axes[1]
ax2.barh([SUB_LABELS[s] for s in fdr2_df.index], partial_coefs.values, color=colors2)
ax2.axvline(0, color="black", lw=0.8)
ax2.set_xlabel("Partial OLS Coefficient")
ax2.set_title("RC ~ IC + 11 Sub-components\n(partial effects beyond IC)")
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(f"{OUT}pattern_regression_coefs.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: pattern_regression_coefs.png")

# --- Fig 2: Story-level heatmap (stories × sub-components, sorted by RC) ---
story_means_sorted = story_means.sort_values(RC_COL, ascending=False)
heatmap_data = story_means_sorted[SUBS]

fig, ax = plt.subplots(figsize=(13, 6))
sns.heatmap(
    heatmap_data,
    ax=ax,
    cmap="RdYlGn",
    center=4,
    vmin=1, vmax=7,
    linewidths=0.4,
    annot=True, fmt=".1f",
    xticklabels=[SUB_LABELS[s] for s in SUBS],
    yticklabels=[f"{idx}\n(RC={story_means_sorted.loc[idx, RC_COL]:.1f})"
                 for idx in story_means_sorted.index]
)
ax.set_title("Story-level Sub-component Means (sorted by RC, N=12 stories)")
plt.tight_layout()
plt.savefig(f"{OUT}pattern_story_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: pattern_story_heatmap.png")

# --- Fig 3: Story-level Spearman correlations with RC ---
fig, ax = plt.subplots(figsize=(9, 4))
colors3 = ["#2ca02c" if r > 0 else "#d62728" for r in story_corr_df["rho"]]
ax.barh([SUB_LABELS[s] for s in story_corr_df["sub"]], story_corr_df["rho"], color=colors3)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Spearman ρ with RC")
ax.set_title("Story-level: Sub-component correlation with RC (N=12 stories)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT}pattern_story_corr.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: pattern_story_corr.png")

# --- Fig 4: Cluster profiles radar/bar ---
fig, axes = plt.subplots(1, best_k, figsize=(5 * best_k, 4), sharey=True)
if best_k == 1:
    axes = [axes]
palette = sns.color_palette("tab10", best_k)
for c, ax in enumerate(axes):
    profile = cluster_profiles.loc[c, SUBS]
    ax.bar([SUB_LABELS[s] for s in SUBS], profile.values, color=palette[c], alpha=0.8)
    ax.set_ylim(1, 7)
    ax.axhline(4, color="gray", lw=0.8, linestyle="--")
    rc_mean = cluster_profiles.loc[c, RC_COL]
    ic_mean = cluster_profiles.loc[c, IC_COL]
    n = int(cluster_profiles.loc[c, "n_raters"])
    ax.set_title(f"Cluster {c}\n(n={n}, IC={ic_mean:.1f}→RC={rc_mean:.1f})")
    ax.tick_params(axis="x", rotation=60)
    ax.set_ylabel("Mean Rating")
fig.suptitle("Rater Cluster Profiles (sub-component means)", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}pattern_cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: pattern_cluster_profiles.png")

# --- Fig 5: Silhouette scores ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(list(sil_scores.keys()), list(sil_scores.values()), "o-", color="#1f77b4")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Silhouette Score")
axes[0].set_title("Optimal k selection")
axes[0].axvline(best_k, color="red", linestyle="--", label=f"best k={best_k}")
axes[0].legend()

axes[1].plot(list(inertias.keys()), list(inertias.values()), "o-", color="#ff7f0e")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Inertia")
axes[1].set_title("Elbow plot")
plt.tight_layout()
plt.savefig(f"{OUT}pattern_cluster_selection.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: pattern_cluster_selection.png")

# --- Fig 6: Rater scatter on PCA, colored by cluster ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
fig, ax = plt.subplots(figsize=(7, 5))
for c in range(best_k):
    mask = df["cluster"] == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {c}",
               alpha=0.7, s=40, color=palette[c])
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
ax.set_title("Rater clusters in PCA space (sub-components)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}pattern_cluster_pca.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: pattern_cluster_pca.png")

# --- Fig 7: IC→RC shift distribution per cluster ---
fig, ax = plt.subplots(figsize=(7, 4))
shift_data = [df[df["cluster"] == c][RC_COL].values - df[df["cluster"] == c][IC_COL].values
              for c in range(best_k)]
ax.boxplot(shift_data, labels=[f"Cluster {c}" for c in range(best_k)], patch_artist=True,
           boxprops=dict(facecolor="lightblue"))
ax.axhline(0, color="red", linestyle="--", lw=0.8)
ax.set_ylabel("RC − IC (shift)")
ax.set_title("IC→RC shift by rater cluster")
plt.tight_layout()
plt.savefig(f"{OUT}pattern_cluster_shift.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: pattern_cluster_shift.png")

print("\n✓ Pattern analysis complete.")
