"""
Phase 1B: Gatekeeper Flip comparison — Human vs LLM.

Produces:
  1. Histogram: human flip distribution (overall)
  2. Heatmap: per-model flip patterns across stories (LLMs + human)
  3. Per-story dot plot: human flip rate vs LLM flip rate
  4. Correlation: do stories that flip humans also flip LLMs?
  5. Statistical tests: chi-square on flip rates, Wilcoxon on magnitudes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# --- Column mapping ---
TOPIC_SHORT = {
    "An advanced AI initiates its own permanent shutdown sequence": "ai_shutdown",
    "A lone worker in a Japanese convenience store at 3:00 AM. The city is silent.": "konbini",
    "A professional thief attempting to crack a high-security safe in a dark room. High tension.": "thief",
}

# --- Load data ---
base = Path(__file__).resolve().parent.parent
human_raw = pd.read_csv(base / "data/human_ratings.csv")
llm = pd.read_csv(base / "data/llm_ratings.csv")

# Standardize human data
human_raw["topic"] = human_raw["TOPIC"].map(TOPIC_SHORT)
human_raw["tone"] = human_raw["TONE"].str.lower()
human_raw["story_id"] = human_raw["topic"] + "_" + human_raw["tone"]
human_raw["flip"] = human_raw["O_Final_Creativity"] - human_raw["O_Initial_Creativity"]
human_raw["flipped"] = (human_raw["flip"] != 0).astype(int)
human_raw["flip_direction"] = human_raw["flip"].apply(
    lambda x: "up" if x > 0 else ("down" if x < 0 else "none")
)

STORY_ORDER = sorted(llm["story_id"].unique())

# =====================================================================
# 1. Human flip distribution — histogram
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1a: Overall flip distribution
flip_vals = human_raw["flip"].values
ax = axes[0]
bins = np.arange(-5.5, 6.5, 1)
ax.hist(flip_vals, bins=bins, color="#3498db", edgecolor="white", alpha=0.8)
ax.axvline(0, color="black", linestyle="--", alpha=0.5)
ax.axvline(flip_vals.mean(), color="#e74c3c", linestyle="-", linewidth=2,
           label=f"Mean = {flip_vals.mean():+.2f}")
ax.set_xlabel("Gatekeeper Flip (RC − IC)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Human Flip Distribution (N=115)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)

# Annotate percentages
n_total = len(flip_vals)
n_up = (flip_vals > 0).sum()
n_down = (flip_vals < 0).sum()
n_zero = (flip_vals == 0).sum()
text = f"Up: {n_up} ({100*n_up/n_total:.0f}%)\nDown: {n_down} ({100*n_down/n_total:.0f}%)\nZero: {n_zero} ({100*n_zero/n_total:.0f}%)"
ax.text(0.97, 0.97, text, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# 1b: Human flip by story
ax = axes[1]
human_story_flip = human_raw.groupby("story_id").agg(
    flip_mean=("flip", "mean"),
    flip_rate=("flipped", "mean"),
    n_up=("flip_direction", lambda x: (x == "up").sum()),
    n_down=("flip_direction", lambda x: (x == "down").sum()),
    n_zero=("flip_direction", lambda x: (x == "none").sum()),
    n=("flip", "count"),
).reindex(STORY_ORDER)

x = np.arange(len(STORY_ORDER))
width = 0.6
up_pct = human_story_flip["n_up"] / human_story_flip["n"] * 100
down_pct = human_story_flip["n_down"] / human_story_flip["n"] * 100
zero_pct = human_story_flip["n_zero"] / human_story_flip["n"] * 100

ax.bar(x, up_pct, width, label="Flip up", color="#2ecc71", edgecolor="white")
ax.bar(x, zero_pct, width, bottom=up_pct, label="No flip", color="#bdc3c7", edgecolor="white")
ax.bar(x, down_pct, width, bottom=up_pct + zero_pct, label="Flip down", color="#e74c3c", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([s.replace("_", "\n") for s in STORY_ORDER], fontsize=7, rotation=45, ha="right")
ax.set_ylabel("Percentage", fontsize=11)
ax.set_title("Human Flip Direction by Story", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.set_ylim(0, 100)

fig.tight_layout()
fig.savefig(OUT / "1b_human_flip_distribution.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT / '1b_human_flip_distribution.png'}")

# =====================================================================
# 2. Flip heatmap — models × stories (Fig 3 candidate)
# =====================================================================
# LLM flip per model × story
llm_pivot = llm.pivot_table(index="model_label", columns="story_id", values="gk_flip", aggfunc="first")
llm_pivot = llm_pivot.reindex(columns=STORY_ORDER)

# Human mean flip per story (for comparison row)
human_flip_by_story = human_raw.groupby("story_id")["flip"].mean()

# Order models by total flip count (most flips on top)
model_flip_count = llm.groupby("model_label")["flipped"].sum().sort_values(ascending=True)
llm_pivot = llm_pivot.reindex(model_flip_count.index)

# Add human row
human_row = human_flip_by_story.reindex(STORY_ORDER).values.reshape(1, -1)
full_matrix = np.vstack([llm_pivot.values, human_row])
row_labels = list(llm_pivot.index) + ["HUMAN (mean)"]

fig, ax = plt.subplots(figsize=(14, 9))
cmap = plt.cm.RdBu_r
im = ax.imshow(full_matrix, cmap=cmap, aspect="auto", vmin=-2, vmax=2)

# Annotate cells
for i in range(full_matrix.shape[0]):
    for j in range(full_matrix.shape[1]):
        val = full_matrix[i, j]
        if np.isnan(val):
            continue
        text_color = "white" if abs(val) > 1.2 else "black"
        fmt = f"{val:+.0f}" if i < len(llm_pivot) else f"{val:+.1f}"
        ax.text(j, i, fmt, ha="center", va="center", fontsize=8,
                color=text_color, fontweight="bold" if val != 0 else "normal")

ax.set_xticks(range(len(STORY_ORDER)))
ax.set_xticklabels([s.replace("_", "\n") for s in STORY_ORDER], fontsize=7, rotation=45, ha="right")
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=9)

# Separator line above human row
ax.axhline(len(llm_pivot) - 0.5, color="black", linewidth=2)

cbar = fig.colorbar(im, ax=ax, shrink=0.6, label="Flip magnitude (RC − IC)")
ax.set_title("Gatekeeper Flip: LLM Models vs Human Mean", fontsize=14, fontweight="bold")

fig.tight_layout()
fig.savefig(OUT / "1b_flip_heatmap.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT / '1b_flip_heatmap.png'}")

# =====================================================================
# 3. Per-story comparison: human flip rate vs LLM flip rate
# =====================================================================
llm_story_flip = llm.groupby("story_id").agg(
    llm_flip_rate=("flipped", "mean"),
    llm_flip_mean=("gk_flip", "mean"),
    llm_n=("flipped", "count"),
).reindex(STORY_ORDER)

human_story_flip["human_flip_rate"] = human_story_flip["flip_rate"]
human_story_flip["human_flip_mean"] = human_story_flip["flip_mean"]

comparison = human_story_flip[["human_flip_rate", "human_flip_mean"]].join(
    llm_story_flip[["llm_flip_rate", "llm_flip_mean"]]
)

TONE_COLORS = {"surreal": "#e74c3c", "clinical": "#3498db", "melancholic": "#9b59b6", "witty": "#2ecc71"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 3a: Flip rate scatter
ax = axes[0]
for sid, row in comparison.iterrows():
    tone = sid.rsplit("_", 1)[-1]
    ax.scatter(row["human_flip_rate"] * 100, row["llm_flip_rate"] * 100,
               c=TONE_COLORS[tone], s=100, edgecolors="black", linewidth=0.5, zorder=3)
    ax.annotate(sid.replace("_", "\n"), (row["human_flip_rate"] * 100, row["llm_flip_rate"] * 100),
                fontsize=5, ha="center", va="bottom", xytext=(0, 5), textcoords="offset points")

r_rate = np.corrcoef(comparison["human_flip_rate"], comparison["llm_flip_rate"])[0, 1]
ax.plot([0, 100], [0, 100], "k--", alpha=0.3)
ax.set_xlabel("Human flip rate (%)", fontsize=11)
ax.set_ylabel("LLM flip rate (%)", fontsize=11)
ax.set_title(f"Flip Rate: Human vs LLM  (r = {r_rate:.2f})", fontsize=13, fontweight="bold")
ax.set_xlim(-5, 105)
ax.set_ylim(-5, 105)
ax.grid(True, alpha=0.2)

# 3b: Flip magnitude scatter
ax = axes[1]
for sid, row in comparison.iterrows():
    tone = sid.rsplit("_", 1)[-1]
    ax.scatter(row["human_flip_mean"], row["llm_flip_mean"],
               c=TONE_COLORS[tone], s=100, edgecolors="black", linewidth=0.5, zorder=3)

r_mag = np.corrcoef(comparison["human_flip_mean"], comparison["llm_flip_mean"])[0, 1]
ax.axhline(0, color="grey", linestyle="--", alpha=0.3)
ax.axvline(0, color="grey", linestyle="--", alpha=0.3)
ax.set_xlabel("Human mean flip", fontsize=11)
ax.set_ylabel("LLM mean flip", fontsize=11)
ax.set_title(f"Flip Magnitude: Human vs LLM  (r = {r_mag:.2f})", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.2)

# Legend
tone_handles = [mpatches.Patch(color=c, label=t.capitalize()) for t, c in TONE_COLORS.items()]
fig.legend(handles=tone_handles, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Do Stories That Flip Humans Also Flip LLMs?", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "1b_flip_rate_comparison.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT / '1b_flip_rate_comparison.png'}")

# =====================================================================
# 4. LLM per-model flip distribution (paired with human)
# =====================================================================
# For each model that flips, show its flip distribution alongside human
flipping_models = llm.groupby("model_label")["flipped"].sum()
flipping_models = flipping_models[flipping_models > 0].sort_values(ascending=False)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flat

# First panel: human
ax = axes[0]
bins = np.arange(-5.5, 6.5, 1)
ax.hist(human_raw["flip"], bins=bins, color="#e67e22", edgecolor="white", alpha=0.8)
ax.axvline(0, color="black", linestyle="--", alpha=0.5)
ax.set_title(f"HUMAN (N={len(human_raw)})", fontsize=11, fontweight="bold")
ax.set_xlabel("Flip (RC−IC)")
ax.set_xlim(-5.5, 5.5)
n = len(human_raw)
rate = human_raw["flipped"].mean()
ax.text(0.97, 0.97, f"Flip rate: {rate:.0%}\nMean: {human_raw['flip'].mean():+.2f}",
        transform=ax.transAxes, fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# Model panels
for idx, (model, _) in enumerate(flipping_models.items()):
    if idx >= 5:
        break
    ax = axes[idx + 1]
    model_data = llm[llm["model_label"] == model]
    flips = model_data["gk_flip"].values
    ax.hist(flips, bins=np.arange(-2.5, 3.5, 1), color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    rate = model_data["flipped"].mean()
    mean_flip = flips.mean()
    ax.set_title(f"{model} (N=12)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Flip (RC−IC)")
    ax.set_xlim(-2.5, 2.5)
    ax.text(0.97, 0.97, f"Flip rate: {rate:.0%}\nMean: {mean_flip:+.2f}",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

# Hide unused axes
for idx in range(len(flipping_models) + 1, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle("Flip Distributions: Human vs Flipping LLM Models", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "1b_flip_distributions.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT / '1b_flip_distributions.png'}")

# =====================================================================
# 5. Statistical tests
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 1B: GATEKEEPER FLIP STATISTICS")
print("=" * 70)

# --- Human flip stats ---
print("\n--- Human Flip Summary ---")
print(f"  N = {len(human_raw)}")
print(f"  Mean flip: {human_raw['flip'].mean():+.3f} (SD = {human_raw['flip'].std():.3f})")
print(f"  Median flip: {human_raw['flip'].median():+.1f}")
print(f"  Flip rate: {human_raw['flipped'].mean():.1%} ({human_raw['flipped'].sum()}/{len(human_raw)})")
print(f"  Up: {(human_raw['flip'] > 0).sum()} ({(human_raw['flip'] > 0).mean():.1%})")
print(f"  Down: {(human_raw['flip'] < 0).sum()} ({(human_raw['flip'] < 0).mean():.1%})")
print(f"  Zero: {(human_raw['flip'] == 0).sum()} ({(human_raw['flip'] == 0).mean():.1%})")

# One-sample Wilcoxon: is human mean flip different from 0?
stat_w, p_w = stats.wilcoxon(human_raw["flip"][human_raw["flip"] != 0])
print(f"  Wilcoxon signed-rank (flip ≠ 0): W={stat_w:.0f}, p={p_w:.4f}")

# --- LLM flip stats per model ---
print("\n--- LLM Flip Summary (per model) ---")
model_summary = llm.groupby("model_label").agg(
    family=("family", "first"),
    flip_rate=("flipped", "mean"),
    n_flips=("flipped", "sum"),
    mean_flip=("gk_flip", "mean"),
    n_up=("flip_direction", lambda x: (x == "up").sum()),
    n_down=("flip_direction", lambda x: (x == "down").sum()),
).sort_values("flip_rate", ascending=False)

for model, row in model_summary.iterrows():
    print(f"  {model:35s} [{row['family']:10s}]  rate={row['flip_rate']:.0%}  "
          f"({row['n_flips']:.0f}/12)  mean={row['mean_flip']:+.2f}  "
          f"↑{row['n_up']:.0f} ↓{row['n_down']:.0f}")

# --- Chi-square: Human flip rate vs each LLM model ---
print("\n--- Chi-square: Human vs LLM Flip Rates ---")
human_flip_count = human_raw["flipped"].sum()
human_no_flip = len(human_raw) - human_flip_count

for model, row in model_summary.iterrows():
    llm_flip = int(row["n_flips"])
    llm_no_flip = 12 - llm_flip
    # Contingency table: [[human_flip, human_noflip], [llm_flip, llm_noflip]]
    table = np.array([[human_flip_count, human_no_flip], [llm_flip, llm_no_flip]])
    # Use Fisher's exact for small LLM N
    odds, p_fisher = stats.fisher_exact(table)
    sig = "***" if p_fisher < 0.001 else "**" if p_fisher < 0.01 else "*" if p_fisher < 0.05 else ""
    print(f"  Human vs {model:35s}  Fisher p={p_fisher:.4f} {sig}  "
          f"(human={human_raw['flipped'].mean():.0%} vs llm={row['flip_rate']:.0%})")

# --- Wilcoxon: human flip magnitude vs pooled LLM flip magnitude per story ---
print("\n--- Per-Story Flip Comparison (human mean vs LLM mean) ---")
for sid in STORY_ORDER:
    h = human_raw[human_raw["story_id"] == sid]["flip"]
    l = llm[llm["story_id"] == sid]["gk_flip"]
    print(f"  {sid:30s}  human={h.mean():+.2f} (SD={h.std():.2f}, N={len(h)})  "
          f"llm={l.mean():+.2f} (SD={l.std():.2f}, N={len(l)})")

# --- Correlation: story-level human flip rate vs LLM flip rate ---
print("\n--- Story-Level Correlations ---")
print(f"  Flip rate (human vs LLM):      r = {r_rate:+.3f}")
print(f"  Flip magnitude (human vs LLM): r = {r_mag:+.3f}")

# Spearman rank correlation
rho_rate, p_rho_rate = stats.spearmanr(comparison["human_flip_rate"], comparison["llm_flip_rate"])
rho_mag, p_rho_mag = stats.spearmanr(comparison["human_flip_mean"], comparison["llm_flip_mean"])
print(f"  Flip rate Spearman:     ρ = {rho_rate:+.3f}, p = {p_rho_rate:.4f}")
print(f"  Flip magnitude Spearman: ρ = {rho_mag:+.3f}, p = {p_rho_mag:.4f}")

# --- Overall comparison: human vs pooled LLM ---
print("\n--- Overall: Human vs Pooled LLM ---")
print(f"  Human flip rate:   {human_raw['flipped'].mean():.1%}")
print(f"  LLM flip rate:     {llm['flipped'].mean():.1%}")
print(f"  Human mean flip:   {human_raw['flip'].mean():+.3f}")
print(f"  LLM mean flip:     {llm['gk_flip'].mean():+.3f}")
print(f"  Human flip SD:     {human_raw['flip'].std():.3f}")
print(f"  LLM flip SD:       {llm['gk_flip'].std():.3f}")

# Mann-Whitney U: human flips vs LLM flips (unpaired)
u_stat, p_mw = stats.mannwhitneyu(human_raw["flip"], llm["gk_flip"], alternative="two-sided")
print(f"  Mann-Whitney U (human vs LLM flips): U={u_stat:.0f}, p={p_mw:.4f}")

print("\n" + "=" * 70)
print("Phase 1B complete. Figures saved to study1_human/analysis/figures/")
print("=" * 70)
