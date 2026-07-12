"""
Generate feature importance comparison figure: Decision Tree for
Initial Creativity vs Reflective Creativity.

Reproduces the "Gatekeeper Flip" figure from the January submission,
showing how evaluation order reverses between IC and RC.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor

base = Path(__file__).resolve().parent.parent
OUT = base / "analysis/figures"
OUT.mkdir(exist_ok=True)

# Column mapping
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

# Aggregate to 4 parent dimensions
DIMENSIONS = {
    "Adherence": ["topic_fidelity", "tone_fidelity"],
    "Novelty": ["vocabulary_freshness", "plot_uniqueness", "surprise"],
    "Technical Value": ["engagement", "stylistic_quality", "logical_coherence"],
    "Resonance": ["emotional_impact", "thought_provocation", "empathy"],
}

DIM_COLORS = {
    "Adherence": "#3498db",
    "Novelty": "#e74c3c",
    "Technical Value": "#2ecc71",
    "Resonance": "#9b59b6",
}

human_raw = pd.read_csv(base / "data/human_ratings.csv").rename(columns=HUMAN_TO_LLM)

# Compute parent dimension means
for dim, cols in DIMENSIONS.items():
    human_raw[dim] = human_raw[cols].mean(axis=1)

dim_names = list(DIMENSIONS.keys())
X = human_raw[dim_names].values

# Decision Tree for IC
dt_ic = DecisionTreeRegressor(max_depth=3, min_samples_split=10, random_state=42)
dt_ic.fit(X, human_raw["IC"].values)
imp_ic = dict(zip(dim_names, dt_ic.feature_importances_))

# Decision Tree for RC
dt_rc = DecisionTreeRegressor(max_depth=3, min_samples_split=10, random_state=42)
dt_rc.fit(X, human_raw["RC"].values)
imp_rc = dict(zip(dim_names, dt_rc.feature_importances_))

print("Feature Importances (Decision Tree, depth=3):")
print(f"  {'Dimension':20s} {'IC':>8s} {'RC':>8s} {'Shift':>8s}")
for d in dim_names:
    print(f"  {d:20s} {imp_ic[d]:8.3f} {imp_rc[d]:8.3f} {imp_rc[d]-imp_ic[d]:+8.3f}")

# R² scores
from sklearn.model_selection import cross_val_score
cv_ic = cross_val_score(DecisionTreeRegressor(max_depth=3, min_samples_split=10, random_state=42),
                        X, human_raw["IC"].values, cv=5, scoring="r2")
cv_rc = cross_val_score(DecisionTreeRegressor(max_depth=3, min_samples_split=10, random_state=42),
                        X, human_raw["RC"].values, cv=5, scoring="r2")
print(f"\n  IC Decision Tree: R² = {cv_ic.mean():.3f} (±{cv_ic.std():.3f})")
print(f"  RC Decision Tree: R² = {cv_rc.mean():.3f} (±{cv_rc.std():.3f})")

# --- Figure ---
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(dim_names))
width = 0.35

ic_vals = [imp_ic[d] for d in dim_names]
rc_vals = [imp_rc[d] for d in dim_names]
colors = [DIM_COLORS[d] for d in dim_names]

bars_ic = ax.bar(x - width/2, ic_vals, width, label="Initial Creativity (IC)",
                 color=[c + "88" for c in colors], edgecolor=colors, linewidth=1.5)
bars_rc = ax.bar(x + width/2, rc_vals, width, label="Reflective Creativity (RC)",
                 color=colors, edgecolor="black", linewidth=0.5)

# Add value labels
for i, (v_ic, v_rc) in enumerate(zip(ic_vals, rc_vals)):
    ax.text(i - width/2, v_ic + 0.01, f"{v_ic:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(i + width/2, v_rc + 0.01, f"{v_rc:.2f}", ha="center", va="bottom", fontsize=9)

# Arrows showing shift direction
for i, d in enumerate(dim_names):
    shift = imp_rc[d] - imp_ic[d]
    if abs(shift) > 0.05:
        arrow_color = "#2ecc71" if shift > 0 else "#e74c3c"
        ax.annotate("", xy=(i + width/2, rc_vals[i]),
                     xytext=(i - width/2, ic_vals[i]),
                     arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5))

ax.set_xticks(x)
ax.set_xticklabels(dim_names, fontsize=11)
ax.set_ylabel("Feature Importance (Gini)", fontsize=12)
ax.set_title("Reversal in Evaluation Order:\nInitial vs Reflective Creativity",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.set_ylim(0, max(max(ic_vals), max(rc_vals)) * 1.2)
ax.grid(True, axis="y", alpha=0.2)

fig.tight_layout()
outpath = OUT / "feature_importance_ic_vs_rc.png"
fig.savefig(outpath, dpi=200, bbox_inches="tight")
# Also save to main figures directory
fig.savefig(base.parent / "figures" / "feature_importance_ic_vs_rc.png", dpi=200, bbox_inches="tight")
print(f"\nSaved: {outpath}")
