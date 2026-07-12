"""
Phase 1A: Story-level human vs LLM profile comparison.

Produces:
  1. Scatter plots: human mean vs LLM mean for IC, RC, enjoyment (per story)
  2. Radar plots: human vs LLM sub-component profiles per tone
  3. Summary stats CSV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# --- Column mapping: human → LLM ---
HUMAN_TO_LLM = {
    "O_Initial_Creativity": "IC",
    "O_Final_Creativity": "RC",
    "O_Enjoyment": "enjoyment",
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
human_raw = human_raw.rename(columns=HUMAN_TO_LLM)

# --- Aggregate to story level ---
shared_cols = ["IC", "RC", "enjoyment"] + SUB_COMPONENTS

human_story = human_raw.groupby("story_id")[shared_cols].agg(["mean", "std", "count"])
human_story.columns = [f"{col}_{stat}" for col, stat in human_story.columns]
human_story = human_story.reset_index()

llm_story = llm.groupby("story_id")[shared_cols].agg(["mean", "std", "count"])
llm_story.columns = [f"{col}_{stat}" for col, stat in llm_story.columns]
llm_story = llm_story.reset_index()

# Merge
merged = human_story.merge(llm_story, on="story_id", suffixes=("_human", "_llm"))

# Extract tone/topic for coloring
merged["tone"] = merged["story_id"].str.rsplit("_", n=1).str[-1]
merged["topic"] = merged["story_id"].str.rsplit("_", n=1).str[0]

# --- Save summary CSV ---
summary_cols = ["story_id", "tone", "topic"]
for col in shared_cols:
    summary_cols += [f"{col}_mean_human", f"{col}_std_human", f"{col}_mean_llm", f"{col}_std_llm"]
merged.to_csv(base / "analysis/phase1a_summary.csv", index=False)
print(f"Summary saved: {base / 'analysis/phase1a_summary.csv'}")

# --- Plot 1: Scatter plots for IC, RC, enjoyment ---
TONE_COLORS = {"surreal": "#e74c3c", "clinical": "#3498db", "melancholic": "#9b59b6", "witty": "#2ecc71"}
TOPIC_MARKERS = {"ai_shutdown": "o", "konbini": "s", "thief": "D"}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, measure in zip(axes, ["IC", "RC", "enjoyment"]):
    hcol = f"{measure}_mean_human"
    lcol = f"{measure}_mean_llm"

    for _, row in merged.iterrows():
        ax.scatter(row[hcol], row[lcol],
                   c=TONE_COLORS[row["tone"]],
                   marker=TOPIC_MARKERS[row["topic"]],
                   s=100, edgecolors="black", linewidth=0.5, zorder=3)

    # Correlation
    r = np.corrcoef(merged[hcol], merged[lcol])[0, 1]
    ax.set_title(f"{measure}  (r = {r:.2f})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Human mean", fontsize=11)
    ax.set_ylabel("LLM mean", fontsize=11)

    # Diagonal
    lims = [1, 7]
    ax.plot(lims, lims, "k--", alpha=0.3, zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

# Legend
from matplotlib.lines import Line2D
tone_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=t.capitalize())
                for t, c in TONE_COLORS.items()]
topic_handles = [Line2D([0], [0], marker=m, color="w", markerfacecolor="grey", markersize=10, label=t)
                 for t, m in TOPIC_MARKERS.items()]
fig.legend(handles=tone_handles + topic_handles, loc="lower center", ncol=7, fontsize=9,
           bbox_to_anchor=(0.5, -0.05))

fig.suptitle("Human vs LLM Story-Level Ratings", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "1a_scatter_human_vs_llm.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT / '1a_scatter_human_vs_llm.png'}")

# --- Plot 2: Radar plots — human vs LLM sub-component profiles by tone ---
tones = ["surreal", "clinical", "melancholic", "witty"]
angles = np.linspace(0, 2 * np.pi, len(SUB_COMPONENTS), endpoint=False).tolist()
angles += angles[:1]  # close the polygon

# Short labels for radar
SHORT_LABELS = ["emotion", "topic", "vocab", "plot", "surprise",
                "empathy", "thought", "engage", "style", "logic", "tone"]

fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(polar=True))

for ax, tone in zip(axes.flat, tones):
    # Human means for this tone
    tone_stories = merged[merged["tone"] == tone]
    h_vals = [tone_stories[f"{sc}_mean_human"].mean() for sc in SUB_COMPONENTS]
    l_vals = [tone_stories[f"{sc}_mean_llm"].mean() for sc in SUB_COMPONENTS]
    h_vals += h_vals[:1]
    l_vals += l_vals[:1]

    ax.plot(angles, h_vals, "o-", color="#e67e22", linewidth=2, label="Human", markersize=5)
    ax.fill(angles, h_vals, color="#e67e22", alpha=0.1)
    ax.plot(angles, l_vals, "s-", color="#2980b9", linewidth=2, label="LLM", markersize=5)
    ax.fill(angles, l_vals, color="#2980b9", alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(SHORT_LABELS, fontsize=8)
    ax.set_ylim(0, 7)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_yticklabels(["1", "2", "3", "4", "5", "6", "7"], fontsize=7)
    ax.set_title(tone.capitalize(), fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

fig.suptitle("Sub-Component Profiles: Human vs LLM (by Tone)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "1a_radar_human_vs_llm.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT / '1a_radar_human_vs_llm.png'}")

# --- Print key stats ---
print("\n=== Correlation Summary (human mean vs LLM mean, per story) ===")
for col in shared_cols:
    hc = f"{col}_mean_human"
    lc = f"{col}_mean_llm"
    r = np.corrcoef(merged[hc], merged[lc])[0, 1]
    h_mean = merged[hc].mean()
    l_mean = merged[lc].mean()
    print(f"  {col:25s}  r={r:+.2f}  human={h_mean:.2f}  llm={l_mean:.2f}  delta={l_mean - h_mean:+.2f}")

print("\n=== Human Variance vs LLM Variance (mean SD across stories) ===")
for col in ["IC", "RC", "enjoyment"] + SUB_COMPONENTS[:5]:
    h_sd = merged[f"{col}_std_human"].mean()
    l_sd = merged[f"{col}_std_llm"].mean()
    print(f"  {col:25s}  human_SD={h_sd:.2f}  llm_SD={l_sd:.2f}  ratio={l_sd/h_sd:.2f}")
