"""
Phase 1D: Variance & Agreement comparison — Human vs LLM.

Computes:
  1. Per-story SD: human inter-rater vs LLM inter-model
  2. Krippendorff's alpha: human inter-rater agreement on IC, RC
  3. LLM inter-model agreement: same metric
  4. Coefficient of variation per measure
  5. Figures: SD comparison, agreement summary

Produces:
  1. Paired bar: human SD vs LLM SD per story
  2. CV comparison across all measures
  3. Agreement summary table (printed + figure)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

TOPIC_SHORT = {
    "An advanced AI initiates its own permanent shutdown sequence": "ai_shutdown",
    "A lone worker in a Japanese convenience store at 3:00 AM. The city is silent.": "konbini",
    "A professional thief attempting to crack a high-security safe in a dark room. High tension.": "thief",
}

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

MEASURES = ["IC", "RC", "enjoyment"]
SUB_COMPONENTS = [
    "emotional_impact", "topic_fidelity", "vocabulary_freshness",
    "plot_uniqueness", "surprise", "empathy", "thought_provocation",
    "engagement", "stylistic_quality", "logical_coherence", "tone_fidelity",
]
ALL_COLS = MEASURES + SUB_COMPONENTS

# --- Load data ---
base = Path(__file__).resolve().parent.parent
human_raw = pd.read_csv(base / "data/human_ratings.csv")
llm = pd.read_csv(base / "data/llm_ratings.csv")

human_raw["topic"] = human_raw["TOPIC"].map(TOPIC_SHORT)
human_raw["tone"] = human_raw["TONE"].str.lower()
human_raw["story_id"] = human_raw["topic"] + "_" + human_raw["tone"]
human_raw = human_raw.rename(columns=HUMAN_TO_LLM)

STORY_ORDER = sorted(llm["story_id"].unique())


# =====================================================================
# Krippendorff's alpha (ordinal)
# =====================================================================
def krippendorff_alpha(ratings_matrix, level="ordinal"):
    """
    Compute Krippendorff's alpha for reliability.
    ratings_matrix: shape (n_items, n_raters), NaN for missing.
    """
    # Remove items with < 2 ratings
    valid = np.sum(~np.isnan(ratings_matrix), axis=1) >= 2
    R = ratings_matrix[valid]
    n_items, n_raters = R.shape

    # All observed values
    all_vals = R[~np.isnan(R)]
    if len(all_vals) < 2:
        return np.nan

    # Observed disagreement
    Do = 0.0
    n_pairs = 0
    for i in range(n_items):
        vals = R[i, ~np.isnan(R[i])]
        m = len(vals)
        if m < 2:
            continue
        for a, b in combinations(vals, 2):
            if level == "ordinal":
                Do += (a - b) ** 2
            else:  # nominal
                Do += 0 if a == b else 1
            n_pairs += 1

    if n_pairs == 0:
        return np.nan
    Do /= n_pairs

    # Expected disagreement
    De = 0.0
    n_all_pairs = 0
    all_vals_list = list(all_vals)
    # For efficiency, compute from value distribution
    if level == "ordinal":
        n_total = len(all_vals_list)
        mean_val = np.mean(all_vals_list)
        var_val = np.var(all_vals_list)
        De = var_val * (n_total + 1) / (n_total - 1) if n_total > 1 else 0
        # More precise: use all pairs from pooled data
        De = 0.0
        for a, b in combinations(all_vals_list, 2):
            De += (a - b) ** 2
            n_all_pairs += 1
        De /= n_all_pairs if n_all_pairs > 0 else 1
    else:
        for a, b in combinations(all_vals_list, 2):
            De += 0 if a == b else 1
            n_all_pairs += 1
        De /= n_all_pairs if n_all_pairs > 0 else 1

    if De == 0:
        return 1.0
    return 1.0 - Do / De


def build_ratings_matrix(df, story_col, value_col, rater_col=None):
    """Build items × raters matrix. If no rater_col, use row order within story."""
    stories = sorted(df[story_col].unique())
    groups = df.groupby(story_col)
    max_raters = max(len(g) for _, g in groups)

    matrix = np.full((len(stories), max_raters), np.nan)
    for i, sid in enumerate(stories):
        vals = df[df[story_col] == sid][value_col].values
        matrix[i, :len(vals)] = vals

    return matrix


# =====================================================================
# 1. Per-story SD comparison
# =====================================================================
human_sd = human_raw.groupby("story_id")[ALL_COLS].std().reindex(STORY_ORDER)
llm_sd = llm.groupby("story_id")[ALL_COLS].std().reindex(STORY_ORDER)

# Figure 1: SD comparison for IC, RC, enjoyment
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
x = np.arange(len(STORY_ORDER))
width = 0.35

for ax, measure in zip(axes, MEASURES):
    h_vals = human_sd[measure].values
    l_vals = llm_sd[measure].values

    ax.bar(x - width / 2, h_vals, width, label="Human", color="#e67e22", edgecolor="white", alpha=0.85)
    ax.bar(x + width / 2, l_vals, width, label="LLM", color="#2980b9", edgecolor="white", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in STORY_ORDER], fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("SD", fontsize=11)
    ax.set_title(measure, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)

    # Mean SD annotation
    ax.text(0.02, 0.97, f"Mean: H={h_vals.mean():.2f}, L={l_vals.mean():.2f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

fig.suptitle("Inter-Rater (Human) vs Inter-Model (LLM) Variability", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "1d_sd_comparison.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT / '1d_sd_comparison.png'}")

# =====================================================================
# 2. CV comparison across all measures
# =====================================================================
human_cv = (human_raw[ALL_COLS].std() / human_raw[ALL_COLS].mean()).rename("human_CV")
llm_cv = (llm[ALL_COLS].std() / llm[ALL_COLS].mean()).rename("llm_CV")

cv_df = pd.DataFrame({"human_CV": human_cv, "llm_CV": llm_cv})
cv_df["ratio"] = cv_df["llm_CV"] / cv_df["human_CV"]

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(ALL_COLS))
width = 0.35

ax.bar(x - width / 2, cv_df["human_CV"], width, label="Human", color="#e67e22", edgecolor="white", alpha=0.85)
ax.bar(x + width / 2, cv_df["llm_CV"], width, label="LLM", color="#2980b9", edgecolor="white", alpha=0.85)

short = {"IC": "IC", "RC": "RC", "enjoyment": "Enjoy",
         "emotional_impact": "Emotion", "topic_fidelity": "Topic",
         "vocabulary_freshness": "Vocab", "plot_uniqueness": "Plot",
         "surprise": "Surprise", "empathy": "Empathy",
         "thought_provocation": "Thought", "engagement": "Engage",
         "stylistic_quality": "Style", "logical_coherence": "Logic",
         "tone_fidelity": "Tone"}

ax.set_xticks(x)
ax.set_xticklabels([short[c] for c in ALL_COLS], fontsize=9, rotation=30, ha="right")
ax.set_ylabel("Coefficient of Variation", fontsize=11)
ax.set_title("Rating Variability: Human vs LLM (CV = SD/Mean)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig(OUT / "1d_cv_comparison.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT / '1d_cv_comparison.png'}")

# =====================================================================
# 3. Krippendorff's alpha
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 1D: VARIANCE & AGREEMENT")
print("=" * 70)

print("\n--- Krippendorff's Alpha (ordinal) ---")
print(f"  {'Measure':25s} {'Human α':>10s} {'LLM α':>10s} {'Δ':>8s}")

alpha_rows = []
for col in ALL_COLS:
    # Human: items = stories, raters = individual humans
    h_matrix = build_ratings_matrix(human_raw, "story_id", col)
    h_alpha = krippendorff_alpha(h_matrix, level="ordinal")

    # LLM: items = stories, raters = model conditions
    l_matrix = build_ratings_matrix(llm, "story_id", col, rater_col="model_label")
    l_alpha = krippendorff_alpha(l_matrix, level="ordinal")

    delta = l_alpha - h_alpha if not (np.isnan(h_alpha) or np.isnan(l_alpha)) else np.nan
    print(f"  {col:25s} {h_alpha:10.3f} {l_alpha:10.3f} {delta:+8.3f}")
    alpha_rows.append({"measure": col, "human_alpha": h_alpha, "llm_alpha": l_alpha, "delta": delta})

alpha_df = pd.DataFrame(alpha_rows)

# =====================================================================
# 4. Agreement summary figure
# =====================================================================
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(ALL_COLS))
width = 0.35

ax.bar(x - width / 2, alpha_df["human_alpha"], width, label="Human inter-rater",
       color="#e67e22", edgecolor="white", alpha=0.85)
ax.bar(x + width / 2, alpha_df["llm_alpha"], width, label="LLM inter-model",
       color="#2980b9", edgecolor="white", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([short[c] for c in ALL_COLS], fontsize=9, rotation=30, ha="right")
ax.set_ylabel("Krippendorff's α", fontsize=12)
ax.axhline(0.667, color="grey", linestyle="--", alpha=0.5, label="α = 0.667 (acceptable)")
ax.axhline(0.8, color="grey", linestyle=":", alpha=0.5, label="α = 0.8 (good)")
ax.set_title("Inter-Rater Agreement: Human vs LLM", fontsize=14, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, axis="y", alpha=0.2)
ax.set_ylim(-0.15, 1.0)

fig.tight_layout()
fig.savefig(OUT / "1d_agreement_comparison.png", dpi=200, bbox_inches="tight")
print(f"\nSaved: {OUT / '1d_agreement_comparison.png'}")

# =====================================================================
# 5. Per-story SD summary table
# =====================================================================
print("\n--- Mean SD Across Stories ---")
print(f"  {'Measure':25s} {'Human SD':>10s} {'LLM SD':>10s} {'Ratio':>8s}")
for col in ALL_COLS:
    h = human_sd[col].mean()
    l = llm_sd[col].mean()
    ratio = l / h if h > 0 else np.inf
    print(f"  {col:25s} {h:10.3f} {l:10.3f} {ratio:8.2f}")

print(f"\n  Overall mean SD ratio (LLM/Human): "
      f"{llm_sd[ALL_COLS].mean().mean() / human_sd[ALL_COLS].mean().mean():.2f}")

# =====================================================================
# 6. CV summary
# =====================================================================
print("\n--- Coefficient of Variation ---")
print(f"  {'Measure':25s} {'Human CV':>10s} {'LLM CV':>10s} {'Ratio':>8s}")
for col in ALL_COLS:
    print(f"  {col:25s} {cv_df.loc[col, 'human_CV']:10.3f} "
          f"{cv_df.loc[col, 'llm_CV']:10.3f} {cv_df.loc[col, 'ratio']:8.2f}")

print(f"\n  Mean CV ratio (LLM/Human): {cv_df['ratio'].mean():.2f}")

# =====================================================================
# 7. Overall summary
# =====================================================================
print("\n--- Summary ---")
h_alpha_mean = alpha_df["human_alpha"].mean()
l_alpha_mean = alpha_df["llm_alpha"].mean()
print(f"  Mean Krippendorff's α:  Human = {h_alpha_mean:.3f},  LLM = {l_alpha_mean:.3f}")
print(f"  LLMs agree with each other {'MORE' if l_alpha_mean > h_alpha_mean else 'LESS'} "
      f"than humans agree with each other")
print(f"  Mean SD ratio (LLM/Human): {llm_sd[ALL_COLS].mean().mean() / human_sd[ALL_COLS].mean().mean():.2f}")
print(f"  Mean CV ratio (LLM/Human): {cv_df['ratio'].mean():.2f}")

print("\n" + "=" * 70)
print("Phase 1D complete. Figures saved to study1_human/analysis/figures/")
print("=" * 70)
