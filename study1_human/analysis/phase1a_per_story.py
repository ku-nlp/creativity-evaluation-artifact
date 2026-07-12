"""
Phase 1A (cont): Per-story human vs LLM comparison.

Produces:
  1. Heatmap pair: human means vs LLM means (stories × sub-components)
  2. Paired bar chart: IC and RC per story, human vs LLM side by side
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
base = Path(__file__).resolve().parent.parent

# --- Load summary ---
summary = pd.read_csv(base / "analysis/phase1a_summary.csv")

SUB_COMPONENTS = [
    "emotional_impact", "topic_fidelity", "vocabulary_freshness",
    "plot_uniqueness", "surprise", "empathy", "thought_provocation",
    "engagement", "stylistic_quality", "logical_coherence", "tone_fidelity",
]

SHORT_SC = ["emotion", "topic", "vocab", "plot", "surprise",
            "empathy", "thought", "engage", "style", "logic", "tone"]

# Sort stories by tone group
TONE_ORDER = {"surreal": 0, "clinical": 1, "melancholic": 2, "witty": 3}
summary["tone_rank"] = summary["tone"].map(TONE_ORDER)
summary = summary.sort_values(["tone_rank", "topic"]).reset_index(drop=True)
stories = summary["story_id"].tolist()

# --- Plot 1: Side-by-side heatmaps (human vs LLM sub-components) ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7),
                                      gridspec_kw={"width_ratios": [5, 5, 5]})

h_matrix = np.array([[summary.loc[summary["story_id"] == s, f"{sc}_mean_human"].values[0]
                       for sc in SUB_COMPONENTS] for s in stories])
l_matrix = np.array([[summary.loc[summary["story_id"] == s, f"{sc}_mean_llm"].values[0]
                       for sc in SUB_COMPONENTS] for s in stories])
diff_matrix = l_matrix - h_matrix

# Human heatmap
im1 = ax1.imshow(h_matrix, cmap="YlOrRd", vmin=1, vmax=7, aspect="auto")
ax1.set_title("Human Means", fontsize=13, fontweight="bold")
ax1.set_yticks(range(len(stories)))
ax1.set_yticklabels([s.replace("_", " ") for s in stories], fontsize=9)
ax1.set_xticks(range(len(SHORT_SC)))
ax1.set_xticklabels(SHORT_SC, fontsize=8, rotation=45, ha="right")
for i in range(len(stories)):
    for j in range(len(SUB_COMPONENTS)):
        ax1.text(j, i, f"{h_matrix[i, j]:.1f}", ha="center", va="center", fontsize=7,
                 color="white" if h_matrix[i, j] > 5 else "black")

# LLM heatmap
im2 = ax2.imshow(l_matrix, cmap="YlOrRd", vmin=1, vmax=7, aspect="auto")
ax2.set_title("LLM Means", fontsize=13, fontweight="bold")
ax2.set_yticks(range(len(stories)))
ax2.set_yticklabels([s.replace("_", " ") for s in stories], fontsize=9)
ax2.set_xticks(range(len(SHORT_SC)))
ax2.set_xticklabels(SHORT_SC, fontsize=8, rotation=45, ha="right")
for i in range(len(stories)):
    for j in range(len(SUB_COMPONENTS)):
        ax2.text(j, i, f"{l_matrix[i, j]:.1f}", ha="center", va="center", fontsize=7,
                 color="white" if l_matrix[i, j] > 5 else "black")

# Difference heatmap (LLM - Human)
norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=3)
im3 = ax3.imshow(diff_matrix, cmap="RdBu_r", norm=norm, aspect="auto")
ax3.set_title("LLM − Human", fontsize=13, fontweight="bold")
ax3.set_yticks(range(len(stories)))
ax3.set_yticklabels([s.replace("_", " ") for s in stories], fontsize=9)
ax3.set_xticks(range(len(SHORT_SC)))
ax3.set_xticklabels(SHORT_SC, fontsize=8, rotation=45, ha="right")
for i in range(len(stories)):
    for j in range(len(SUB_COMPONENTS)):
        ax3.text(j, i, f"{diff_matrix[i, j]:+.1f}", ha="center", va="center", fontsize=7,
                 color="white" if abs(diff_matrix[i, j]) > 1.5 else "black")

# Colorbars
fig.colorbar(im1, ax=ax1, shrink=0.6, label="Score (1-7)")
fig.colorbar(im2, ax=ax2, shrink=0.6, label="Score (1-7)")
fig.colorbar(im3, ax=ax3, shrink=0.6, label="Δ (LLM − Human)")

# Add tone group separators
for ax in [ax1, ax2, ax3]:
    for y in [2.5, 5.5, 8.5]:  # between tone groups (3 stories each)
        ax.axhline(y, color="black", linewidth=1.5)

fig.suptitle("Per-Story Sub-Component Profiles: Human vs LLM", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "1a_heatmap_per_story.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT / '1a_heatmap_per_story.png'}")

# --- Plot 2: Paired bar chart — IC and RC per story ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

x = np.arange(len(stories))
w = 0.35

TONE_COLORS = {"surreal": "#e74c3c", "clinical": "#3498db", "melancholic": "#9b59b6", "witty": "#2ecc71"}
edge_colors = [TONE_COLORS[summary.loc[i, "tone"]] for i in range(len(stories))]

# IC
h_ic = summary["IC_mean_human"].values
l_ic = summary["IC_mean_llm"].values
h_ic_sd = summary["IC_std_human"].values
l_ic_sd = summary["IC_std_llm"].values

bars1 = ax1.bar(x - w/2, h_ic, w, yerr=h_ic_sd, label="Human", color="#e67e22", alpha=0.8,
                capsize=3, edgecolor="black", linewidth=0.5)
bars2 = ax1.bar(x + w/2, l_ic, w, yerr=l_ic_sd, label="LLM", color="#2980b9", alpha=0.8,
                capsize=3, edgecolor="black", linewidth=0.5)
ax1.set_ylabel("Score (1-7)", fontsize=11)
ax1.set_title("Initial Creativity (IC)", fontsize=13, fontweight="bold")
ax1.set_ylim(0, 8)
ax1.legend(fontsize=10)
ax1.grid(axis="y", alpha=0.2)

# Add tone color strip at bottom
for i, c in enumerate(edge_colors):
    ax1.bar(i, 0.15, 0.9, bottom=0, color=c, alpha=0.5)

# RC
h_rc = summary["RC_mean_human"].values
l_rc = summary["RC_mean_llm"].values
h_rc_sd = summary["RC_std_human"].values
l_rc_sd = summary["RC_std_llm"].values

ax2.bar(x - w/2, h_rc, w, yerr=h_rc_sd, label="Human", color="#e67e22", alpha=0.8,
        capsize=3, edgecolor="black", linewidth=0.5)
ax2.bar(x + w/2, l_rc, w, yerr=l_rc_sd, label="LLM", color="#2980b9", alpha=0.8,
        capsize=3, edgecolor="black", linewidth=0.5)
ax2.set_ylabel("Score (1-7)", fontsize=11)
ax2.set_title("Reflective Creativity (RC)", fontsize=13, fontweight="bold")
ax2.set_ylim(0, 8)
ax2.legend(fontsize=10)
ax2.grid(axis="y", alpha=0.2)

for i, c in enumerate(edge_colors):
    ax2.bar(i, 0.15, 0.9, bottom=0, color=c, alpha=0.5)

ax2.set_xticks(x)
ax2.set_xticklabels([s.replace("_", "\n") for s in stories], fontsize=8, ha="center")

# Tone legend
from matplotlib.patches import Patch
tone_patches = [Patch(facecolor=c, alpha=0.5, label=t.capitalize()) for t, c in TONE_COLORS.items()]
fig.legend(handles=tone_patches, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Per-Story IC & RC: Human vs LLM", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "1a_bars_ic_rc_per_story.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT / '1a_bars_ic_rc_per_story.png'}")

# --- Plot 3: Flip comparison per story ---
fig, ax = plt.subplots(figsize=(14, 6))

h_flip = summary["RC_mean_human"].values - summary["IC_mean_human"].values
l_flip = summary["RC_mean_llm"].values - summary["IC_mean_llm"].values

ax.bar(x - w/2, h_flip, w, label="Human (RC−IC)", color="#e67e22", alpha=0.8,
       edgecolor="black", linewidth=0.5)
ax.bar(x + w/2, l_flip, w, label="LLM (RC−IC)", color="#2980b9", alpha=0.8,
       edgecolor="black", linewidth=0.5)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Mean Flip (RC − IC)", fontsize=11)
ax.set_title("Gatekeeper Flip per Story: Human vs LLM", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([s.replace("_", "\n") for s in stories], fontsize=8, ha="center")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.2)

for i, c in enumerate(edge_colors):
    ax.bar(i, 0.02, 0.9, bottom=ax.get_ylim()[0], color=c, alpha=0.5)

fig.tight_layout()
fig.savefig(OUT / "1a_flip_per_story.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT / '1a_flip_per_story.png'}")

# --- Print per-story table ---
print("\n=== Per-Story Human vs LLM ===")
print(f"{'Story':<30s} {'H_IC':>5s} {'L_IC':>5s} {'H_RC':>5s} {'L_RC':>5s} {'H_flip':>7s} {'L_flip':>7s} {'H_SD':>5s} {'L_SD':>5s}")
print("-" * 95)
for _, row in summary.iterrows():
    sid = row["story_id"]
    hic = row["IC_mean_human"]
    lic = row["IC_mean_llm"]
    hrc = row["RC_mean_human"]
    lrc = row["RC_mean_llm"]
    hflip = hrc - hic
    lflip = lrc - lic
    hsd = row["IC_std_human"]
    lsd = row["IC_std_llm"]
    print(f"{sid:<30s} {hic:5.2f} {lic:5.2f} {hrc:5.2f} {lrc:5.2f} {hflip:+7.2f} {lflip:+7.2f} {hsd:5.2f} {lsd:5.2f}")
