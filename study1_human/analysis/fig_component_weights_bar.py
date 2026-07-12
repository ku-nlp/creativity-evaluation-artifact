"""
Generate the 11-component weights bar chart comparing:
- Initial Creativity (IC)
- Reflective Creativity (RC)
- Enjoyment

Shows the "Empathy Gap" and "Vocabulary Trap" from the January submission.
Aggregates 11 sub-component betas into 4 parent dimension percentages,
plus produces the full 11-component figure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

base = Path(__file__).resolve().parent.parent
OUT = base / "analysis/figures"
OUT.mkdir(exist_ok=True)

HUMAN_TO_LLM = {
    "O_Final_Creativity": "RC",
    "O_Initial_Creativity": "IC",
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

DIM_MAP = {
    "emotional_impact": "Resonance",
    "topic_fidelity": "Adherence",
    "vocabulary_freshness": "Novelty",
    "plot_uniqueness": "Novelty",
    "surprise": "Novelty",
    "empathy": "Resonance",
    "thought_provocation": "Resonance",
    "engagement": "Technical Value",
    "stylistic_quality": "Technical Value",
    "logical_coherence": "Technical Value",
    "tone_fidelity": "Adherence",
}

DIM_COLORS = {
    "Adherence": "#3498db",
    "Novelty": "#e74c3c",
    "Technical Value": "#2ecc71",
    "Resonance": "#9b59b6",
}

human_raw = pd.read_csv(base / "data/human_ratings.csv").rename(columns=HUMAN_TO_LLM)


def run_ols_weights(df, y_col, x_cols):
    """Run OLS, return absolute beta weights as percentage shares."""
    df_clean = df[[y_col] + x_cols].dropna()
    X = df_clean[x_cols].values
    y = df_clean[y_col].values

    scaler = StandardScaler()
    X_z = scaler.fit_transform(X)
    y_z = (y - y.mean()) / y.std()

    reg = LinearRegression(fit_intercept=True)
    reg.fit(X_z, y_z)

    abs_betas = np.abs(reg.coef_)
    total = abs_betas.sum()
    pct = (abs_betas / total * 100) if total > 0 else abs_betas

    return dict(zip(x_cols, pct)), dict(zip(x_cols, reg.coef_))


# Compute weights for IC, RC, Enjoyment
pct_ic, beta_ic = run_ols_weights(human_raw, "IC", SUB_COMPONENTS)
pct_rc, beta_rc = run_ols_weights(human_raw, "RC", SUB_COMPONENTS)
pct_enj, beta_enj = run_ols_weights(human_raw, "enjoyment", SUB_COMPONENTS)

# Print dimension-level aggregates
print("=" * 60)
print("DIMENSION-LEVEL WEIGHT PERCENTAGES")
print("=" * 60)
for target, pct_dict in [("IC", pct_ic), ("RC", pct_rc), ("Enjoyment", pct_enj)]:
    print(f"\n{target}:")
    dim_totals = {}
    for sc, p in pct_dict.items():
        dim = DIM_MAP[sc]
        dim_totals[dim] = dim_totals.get(dim, 0) + p
    for dim in ["Resonance", "Technical Value", "Novelty", "Adherence"]:
        print(f"  {dim:20s}: {dim_totals.get(dim, 0):5.1f}%")

print("\n\nKey comparisons:")
print(f"  Empathy → Creativity (RC): {pct_rc['empathy']:.1f}%")
print(f"  Empathy → Enjoyment:       {pct_enj['empathy']:.1f}%")
print(f"  Vocab → Creativity (RC):   {pct_rc['vocabulary_freshness']:.1f}%")
print(f"  Vocab → Enjoyment:         {pct_enj['vocabulary_freshness']:.1f}%")

# --- Figure: 11-component bar chart ---
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(SUB_COMPONENTS))
width = 0.25

ic_vals = [pct_ic[sc] for sc in SUB_COMPONENTS]
rc_vals = [pct_rc[sc] for sc in SUB_COMPONENTS]
enj_vals = [pct_enj[sc] for sc in SUB_COMPONENTS]

bar_colors = [DIM_COLORS[DIM_MAP[sc]] for sc in SUB_COMPONENTS]

ax.bar(x - width, ic_vals, width, label="Initial Creativity", color=bar_colors, alpha=0.5,
       edgecolor="black", linewidth=0.5)
ax.bar(x, rc_vals, width, label="Reflective Creativity", color=bar_colors, alpha=0.8,
       edgecolor="black", linewidth=0.5)
ax.bar(x + width, enj_vals, width, label="Enjoyment", color=bar_colors, alpha=0.3,
       edgecolor="black", linewidth=0.5, hatch="///")

ax.set_xticks(x)
ax.set_xticklabels([SHORT_LABELS[sc] for sc in SUB_COMPONENTS], fontsize=10, rotation=30, ha="right")
ax.set_ylabel("Weight (% of total |β|)", fontsize=12)
ax.set_title("Sub-Component Weights: Creativity vs Enjoyment\n(The Empathy Gap & Vocabulary Trap)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, axis="y", alpha=0.2)

# Highlight key findings
empathy_idx = SUB_COMPONENTS.index("empathy")
vocab_idx = SUB_COMPONENTS.index("vocabulary_freshness")
ax.annotate("Empathy Gap", xy=(empathy_idx + width, pct_enj[SUB_COMPONENTS[empathy_idx]]),
            xytext=(empathy_idx + 1.5, pct_enj[SUB_COMPONENTS[empathy_idx]] + 3),
            fontsize=9, fontweight="bold", color="#9b59b6",
            arrowprops=dict(arrowstyle="->", color="#9b59b6"))
ax.annotate("Vocabulary Trap", xy=(vocab_idx, pct_rc[SUB_COMPONENTS[vocab_idx]]),
            xytext=(vocab_idx + 1.5, pct_rc[SUB_COMPONENTS[vocab_idx]] + 3),
            fontsize=9, fontweight="bold", color="#e74c3c",
            arrowprops=dict(arrowstyle="->", color="#e74c3c"))

# Dimension legend
import matplotlib.patches as mpatches
dim_handles = [mpatches.Patch(color=c, label=d) for d, c in DIM_COLORS.items()]
ax.legend(handles=ax.get_legend_handles_labels()[1] + dim_handles,
          labels=["Initial Creativity", "Reflective Creativity", "Enjoyment"] + list(DIM_COLORS.keys()),
          fontsize=8, loc="upper right", ncol=2)

# Fix legend
handles1, labels1 = ax.get_legend_handles_labels()
ax.legend(fontsize=8, loc="upper right", ncol=2)

fig.tight_layout()
outpath = OUT / "component_weights_comparison.png"
fig.savefig(outpath, dpi=200, bbox_inches="tight")
fig.savefig(base.parent / "figures" / "component_weights_comparison.png", dpi=200, bbox_inches="tight")
print(f"\nSaved: {outpath}")
