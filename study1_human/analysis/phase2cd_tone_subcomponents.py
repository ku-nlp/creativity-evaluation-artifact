"""
Phase 2C: Tone and story effects (LLM-only)
Phase 2D: Sub-component patterns (LLM-only)

Produces:
  1. Kruskal-Wallis: IC ~ tone, post-hoc pairwise
  2. Story difficulty ranking
  3. Topic effect after controlling for tone
  4. Sub-component correlation matrix
  5. PCA on sub-components
  6. Discriminating dimensions (highest inter-model variance)
  7. Ceiling effect identification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

base = Path(__file__).resolve().parent.parent
llm = pd.read_csv(base / "data/llm_ratings.csv")

SUB_COMPONENTS = [
    "emotional_impact", "topic_fidelity", "vocabulary_freshness",
    "plot_uniqueness", "surprise", "empathy", "thought_provocation",
    "engagement", "stylistic_quality", "logical_coherence", "tone_fidelity",
]
SHORT = {
    "emotional_impact": "Emotion", "topic_fidelity": "Topic",
    "vocabulary_freshness": "Vocab", "plot_uniqueness": "Plot",
    "surprise": "Surprise", "empathy": "Empathy",
    "thought_provocation": "Thought", "engagement": "Engage",
    "stylistic_quality": "Style", "logical_coherence": "Logic",
    "tone_fidelity": "Tone",
}

STORY_ORDER = sorted(llm["story_id"].unique())
TONES = ["surreal", "clinical", "melancholic", "witty"]
TONE_COLORS = {"surreal": "#e74c3c", "clinical": "#3498db", "melancholic": "#9b59b6", "witty": "#2ecc71"}

# =====================================================================
# PHASE 2C: Tone and story effects
# =====================================================================
print("=" * 70)
print("PHASE 2C: TONE AND STORY EFFECTS")
print("=" * 70)

# --- Kruskal-Wallis: IC ~ tone ---
print("\n--- Kruskal-Wallis: measure ~ tone ---")
for measure in ["IC", "RC", "enjoyment", "gk_flip"]:
    groups = [llm[llm["tone"] == t][measure].values for t in TONES]
    h_stat, p_val = stats.kruskal(*groups)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"  {measure:12s}  H={h_stat:.2f}, p={p_val:.4f} {sig}")

    # Tone means
    for t in TONES:
        m = llm[llm["tone"] == t][measure].mean()
        s = llm[llm["tone"] == t][measure].std()
        print(f"    {t:15s}  mean={m:.2f} SD={s:.2f}")

# --- Post-hoc pairwise Mann-Whitney ---
print("\n--- Post-hoc Pairwise (Mann-Whitney U, Bonferroni-corrected) ---")
n_comparisons = len(list(combinations(TONES, 2)))
for measure in ["IC", "RC", "enjoyment"]:
    print(f"  {measure}:")
    for t1, t2 in combinations(TONES, 2):
        v1 = llm[llm["tone"] == t1][measure].values
        v2 = llm[llm["tone"] == t2][measure].values
        u, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
        p_adj = min(p * n_comparisons, 1.0)
        sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else ""
        print(f"    {t1:12s} vs {t2:12s}  U={u:.0f}  p_adj={p_adj:.4f} {sig}  "
              f"({llm[llm['tone']==t1][measure].mean():.2f} vs {llm[llm['tone']==t2][measure].mean():.2f})")

# --- Story difficulty ranking ---
print("\n--- Story Difficulty Ranking (mean IC across 17 models) ---")
story_rank = llm.groupby("story_id").agg(
    IC_mean=("IC", "mean"), IC_sd=("IC", "std"),
    RC_mean=("RC", "mean"), flip_rate=("flipped", "mean"),
).sort_values("IC_mean")

for sid, r in story_rank.iterrows():
    print(f"  {sid:30s}  IC={r['IC_mean']:.2f} (SD={r['IC_sd']:.2f})  "
          f"RC={r['RC_mean']:.2f}  flip={r['flip_rate']:.0%}")

# --- Topic effect (controlling for tone) ---
print("\n--- Topic Effect (Kruskal-Wallis: IC ~ topic, controlling for tone) ---")
topics = llm["topic"].unique()
for measure in ["IC", "RC"]:
    groups = [llm[llm["topic"] == t][measure].values for t in topics]
    h_stat, p_val = stats.kruskal(*groups)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"  {measure}: H={h_stat:.2f}, p={p_val:.4f} {sig}")
    for t in topics:
        print(f"    {t:15s}  mean={llm[llm['topic']==t][measure].mean():.2f}")

    # Per-tone topic comparison
    print(f"  Per-tone breakdown ({measure}):")
    for tone in TONES:
        tone_df = llm[llm["tone"] == tone]
        topic_groups = [tone_df[tone_df["topic"] == t][measure].values for t in topics]
        if all(len(g) > 0 for g in topic_groups):
            h, p = stats.kruskal(*topic_groups)
            means = ', '.join(f'{t}={tone_df[tone_df["topic"]==t][measure].mean():.2f}' for t in topics)
            print(f"    {tone:12s}  H={h:.2f}, p={p:.4f}  ({means})")

# =====================================================================
# Figure 2C: Tone effects
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, measure in zip(axes, ["IC", "RC", "enjoyment"]):
    tone_data = [llm[llm["tone"] == t][measure].values for t in TONES]
    bp = ax.boxplot(tone_data, labels=[t.capitalize() for t in TONES], patch_artist=True)
    for patch, tone in zip(bp["boxes"], TONES):
        patch.set_facecolor(TONE_COLORS[tone])
        patch.set_alpha(0.6)
    ax.set_ylabel(measure, fontsize=11)
    ax.set_title(measure, fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2)

    # Add means
    for i, t in enumerate(TONES):
        m = llm[llm["tone"] == t][measure].mean()
        ax.plot(i + 1, m, "ko", markersize=6, zorder=5)

fig.suptitle("Score Distributions by Tone (LLM, N=204)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "2c_tone_boxplots.png", dpi=200, bbox_inches="tight")
print(f"\nSaved: {OUT / '2c_tone_boxplots.png'}")

# Story difficulty figure
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(story_rank))
colors = [TONE_COLORS[sid.rsplit("_", 1)[-1]] for sid in story_rank.index]
ax.barh(x, story_rank["IC_mean"], xerr=story_rank["IC_sd"], color=colors,
        edgecolor="white", alpha=0.8, capsize=3)
ax.set_yticks(x)
ax.set_yticklabels(story_rank.index, fontsize=9)
ax.set_xlabel("Mean IC (±SD across 17 models)", fontsize=11)
ax.set_title("Story Difficulty Ranking", fontsize=14, fontweight="bold")
ax.grid(True, axis="x", alpha=0.2)
tone_handles = [mpatches.Patch(color=c, label=t.capitalize()) for t, c in TONE_COLORS.items()]
ax.legend(handles=tone_handles, fontsize=9, loc="lower right")

fig.tight_layout()
fig.savefig(OUT / "2c_story_difficulty.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT / '2c_story_difficulty.png'}")

# =====================================================================
# PHASE 2D: Sub-component patterns
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 2D: SUB-COMPONENT PATTERNS")
print("=" * 70)

# --- Correlation matrix ---
sc_data = llm[SUB_COMPONENTS].dropna()
corr_matrix = sc_data.corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
labels = [SHORT[sc] for sc in SUB_COMPONENTS]
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)

for i in range(len(SUB_COMPONENTS)):
    for j in range(len(SUB_COMPONENTS)):
        val = corr_matrix.iloc[i, j]
        color = "white" if abs(val) > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
ax.set_title("Sub-Component Correlation Matrix (N=204)", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "2d_correlation_matrix.png", dpi=200, bbox_inches="tight")
print(f"\nSaved: {OUT / '2d_correlation_matrix.png'}")

# --- PCA ---
scaler = StandardScaler()
# Drop zero/near-zero variance columns for PCA
sc_std = sc_data.std()
pca_cols = [c for c in SUB_COMPONENTS if sc_std[c] > 0.1]
X_scaled = scaler.fit_transform(sc_data[pca_cols])

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("\n--- PCA: Explained Variance ---")
for i, (var, cumvar) in enumerate(zip(pca.explained_variance_ratio_,
                                       np.cumsum(pca.explained_variance_ratio_))):
    print(f"  PC{i+1}: {var:.1%} (cumulative: {cumvar:.1%})")

print("\n--- PCA Loadings (PC1, PC2, PC3) ---")
print(f"  {'Component':12s} {'PC1':>8s} {'PC2':>8s} {'PC3':>8s}")
for j, sc in enumerate(pca_cols):
    print(f"  {SHORT[sc]:12s} {pca.components_[0, j]:+8.3f} "
          f"{pca.components_[1, j]:+8.3f} {pca.components_[2, j]:+8.3f}")

# PCA biplot
fig, ax = plt.subplots(figsize=(10, 8))

# Color by tone
for tone in TONES:
    mask = llm["tone"].values == tone
    idx = np.where(mask)[0]
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], c=TONE_COLORS[tone],
               label=tone.capitalize(), alpha=0.5, s=40, edgecolors="none")

# Loading arrows
scale = 3
for j, sc in enumerate(pca_cols):
    ax.annotate(SHORT[sc],
                xy=(pca.components_[0, j] * scale, pca.components_[1, j] * scale),
                fontsize=9, fontweight="bold", color="black",
                arrowprops=dict(arrowstyle="<-", color="grey", lw=1.5),
                xytext=(pca.components_[0, j] * scale * 1.3, pca.components_[1, j] * scale * 1.3))

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", fontsize=11)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})", fontsize=11)
ax.set_title("PCA Biplot: Sub-Components (colored by tone)", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.axhline(0, color="grey", linewidth=0.5)
ax.axvline(0, color="grey", linewidth=0.5)
ax.grid(True, alpha=0.2)

fig.tight_layout()
fig.savefig(OUT / "2d_pca_biplot.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT / '2d_pca_biplot.png'}")

# --- Discriminating dimensions (highest inter-model variance) ---
print("\n--- Discriminating Dimensions (inter-model SD per story, averaged) ---")
disc = []
for sc in SUB_COMPONENTS:
    story_sds = llm.groupby("story_id")[sc].std()
    disc.append({"component": sc, "mean_sd": story_sds.mean(), "min_sd": story_sds.min(),
                 "max_sd": story_sds.max()})
disc_df = pd.DataFrame(disc).sort_values("mean_sd", ascending=False)

print(f"  {'Component':25s} {'Mean SD':>8s} {'Min SD':>8s} {'Max SD':>8s}")
for _, r in disc_df.iterrows():
    print(f"  {r['component']:25s} {r['mean_sd']:8.3f} {r['min_sd']:8.3f} {r['max_sd']:8.3f}")

# --- Ceiling effects ---
print("\n--- Ceiling Effects (measures with mean > 6.5 and SD < 0.5) ---")
for sc in SUB_COMPONENTS:
    m = llm[sc].mean()
    s = llm[sc].std()
    if m > 6.5 or s < 0.5:
        pct_max = (llm[sc] == 7).mean() * 100
        print(f"  {sc:25s}  mean={m:.2f}  SD={s:.2f}  at_ceiling(=7): {pct_max:.0f}%")

# Discriminating dimensions figure
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(disc_df))
ax.bar(x, disc_df["mean_sd"], color="#2980b9", edgecolor="white", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([SHORT[c] for c in disc_df["component"]], fontsize=10, rotation=30, ha="right")
ax.set_ylabel("Mean inter-model SD (per story)", fontsize=11)
ax.set_title("Which Sub-Components Discriminate Between Models?", fontsize=14, fontweight="bold")
ax.grid(True, axis="y", alpha=0.2)
ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="SD = 0.5 threshold")
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "2d_discriminating_dimensions.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT / '2d_discriminating_dimensions.png'}")

print("\n" + "=" * 70)
print("Phase 2C+2D complete.")
print("=" * 70)
