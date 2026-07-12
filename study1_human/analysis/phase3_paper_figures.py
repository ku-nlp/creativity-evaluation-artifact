"""
Phase 3: Paper-ready figures (6 figures for the ARR submission).

All figures use consistent styling:
  - Seaborn whitegrid theme
  - Consistent color palettes
  - Publication-quality DPI (300)
  - PDF + PNG output
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# --- Setup ---
OUT = Path(__file__).parent / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)
base = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# --- Colors ---
HUMAN_COLOR = "#D4740E"
LLM_COLOR = "#1A6FAA"
TONE_COLORS = {"surreal": "#E45756", "clinical": "#4C78A8", "melancholic": "#9D69A3", "witty": "#54A24B"}
TOPIC_MARKERS = {"ai_shutdown": "o", "konbini": "s", "thief": "D"}
FAMILY_COLORS = {"Meta": "#E45756", "Google": "#4C78A8", "Alibaba": "#54A24B",
                 "OpenAI": "#9D69A3", "Microsoft": "#EECA3B"}

# --- Column mapping ---
HUMAN_TO_LLM = {
    "O_Initial_Creativity": "IC", "O_Final_Creativity": "RC", "O_Enjoyment": "enjoyment",
    "R_Emotion": "emotional_impact", "A_Topic": "topic_fidelity",
    "N_Vocab": "vocabulary_freshness", "N_Plot": "plot_uniqueness",
    "N_Surprise": "surprise", "R_Empathy": "empathy", "R_Thought": "thought_provocation",
    "V_Engagement": "engagement", "V_Style": "stylistic_quality",
    "V_Logic": "logical_coherence", "A_Tone": "tone_fidelity",
}
TOPIC_SHORT = {
    "An advanced AI initiates its own permanent shutdown sequence": "ai_shutdown",
    "A lone worker in a Japanese convenience store at 3:00 AM. The city is silent.": "konbini",
    "A professional thief attempting to crack a high-security safe in a dark room. High tension.": "thief",
}
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

# --- Load data ---
human_raw = pd.read_csv(base / "data/human_ratings.csv")
human_raw["topic"] = human_raw["TOPIC"].map(TOPIC_SHORT)
human_raw["tone"] = human_raw["TONE"].str.lower()
human_raw["story_id"] = human_raw["topic"] + "_" + human_raw["tone"]
human_raw = human_raw.rename(columns=HUMAN_TO_LLM)
human_raw["flip"] = human_raw["RC"] - human_raw["IC"]

llm = pd.read_csv(base / "data/llm_ratings.csv")

shared_cols = ["IC", "RC", "enjoyment"] + SUB_COMPONENTS
STORY_ORDER = sorted(llm["story_id"].unique())

# Story-level aggregates
human_story = human_raw.groupby("story_id")[shared_cols].agg(["mean", "std"]).reset_index()
human_story.columns = ["story_id"] + [f"{c}_{s}" for c, s in human_story.columns[1:]]
llm_story = llm.groupby("story_id")[shared_cols].agg(["mean", "std"]).reset_index()
llm_story.columns = ["story_id"] + [f"{c}_{s}" for c, s in llm_story.columns[1:]]
merged = human_story.merge(llm_story, on="story_id", suffixes=("_human", "_llm"))
merged["tone"] = merged["story_id"].str.rsplit("_", n=1).str[-1]
merged["topic"] = merged["story_id"].str.rsplit("_", n=1).str[0]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  Saved: {name}")


# =====================================================================
# FIG 1: Human vs LLM scatter — IC, RC, enjoyment
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

for ax, measure, title in zip(axes, ["IC", "RC", "enjoyment"],
                                ["Initial Creativity", "Reflective Creativity", "Enjoyment"]):
    hcol = f"{measure}_mean_human"
    lcol = f"{measure}_mean_llm"

    for _, row in merged.iterrows():
        ax.scatter(row[hcol], row[lcol],
                   c=TONE_COLORS[row["tone"]],
                   marker=TOPIC_MARKERS[row["topic"]],
                   s=90, edgecolors="black", linewidth=0.5, zorder=3)

    r = np.corrcoef(merged[hcol], merged[lcol])[0, 1]
    ax.set_title(f"{title}  (r = {r:.2f})", fontweight="bold")
    ax.set_xlabel("Human mean")
    ax.set_ylabel("LLM mean")
    ax.plot([1, 7], [1, 7], "k--", alpha=0.25, zorder=1)
    ax.set_xlim(2.5, 7.5)
    ax.set_ylim(2.5, 7.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)

tone_h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8,
                  markeredgecolor="k", markeredgewidth=0.5, label=t.capitalize())
           for t, c in TONE_COLORS.items()]
topic_h = [Line2D([0], [0], marker=m, color="w", markerfacecolor="grey", markersize=8,
                   markeredgecolor="k", markeredgewidth=0.5, label=t.replace("_", " "))
            for t, m in TOPIC_MARKERS.items()]
fig.legend(handles=tone_h + topic_h, loc="lower center", ncol=7, frameon=True,
           bbox_to_anchor=(0.5, -0.08))
fig.tight_layout(rect=[0, 0.06, 1, 1])
save(fig, "fig1_scatter")

# =====================================================================
# FIG 2: Strip plot — best topic (ai_shutdown, highest correlations)
# =====================================================================
topic = "ai_shutdown"
tones = ["surreal", "clinical", "melancholic", "witty"]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, measure, title in zip(axes, ["IC", "RC"], ["Initial Creativity", "Reflective Creativity"]):
    for i, tone in enumerate(tones):
        sid = f"{topic}_{tone}"
        # Human dots
        h_vals = human_raw[human_raw["story_id"] == sid][measure].values
        jitter_h = np.random.default_rng(42).uniform(-0.15, -0.02, len(h_vals))
        ax.scatter(np.full(len(h_vals), i) + jitter_h, h_vals,
                   c=HUMAN_COLOR, alpha=0.5, s=30, edgecolors="none", zorder=2)
        # Human mean
        ax.plot(i - 0.08, h_vals.mean(), "D", color=HUMAN_COLOR, markersize=8,
                markeredgecolor="black", markeredgewidth=0.8, zorder=4)

        # LLM markers (one per model)
        l_vals = llm[llm["story_id"] == sid][measure].values
        jitter_l = np.random.default_rng(42).uniform(0.02, 0.15, len(l_vals))
        ax.scatter(np.full(len(l_vals), i) + jitter_l, l_vals,
                   c=LLM_COLOR, alpha=0.5, s=30, marker="s", edgecolors="none", zorder=2)
        # LLM mean
        ax.plot(i + 0.08, l_vals.mean(), "s", color=LLM_COLOR, markersize=8,
                markeredgecolor="black", markeredgewidth=0.8, zorder=4)

    ax.set_xticks(range(len(tones)))
    ax.set_xticklabels([t.capitalize() for t in tones])
    ax.set_ylabel(measure)
    ax.set_title(f"{title} — AI Shutdown Stories", fontweight="bold")
    ax.set_ylim(0.5, 7.5)
    ax.grid(True, axis="y", alpha=0.15)

legend_h = [Line2D([0], [0], marker="D", color="w", markerfacecolor=HUMAN_COLOR,
                    markersize=8, markeredgecolor="k", label="Human mean"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=HUMAN_COLOR,
                    markersize=6, alpha=0.5, label="Human individual"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor=LLM_COLOR,
                    markersize=8, markeredgecolor="k", label="LLM mean"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor=LLM_COLOR,
                    markersize=6, alpha=0.5, label="LLM individual")]
fig.legend(handles=legend_h, loc="lower center", ncol=4, frameon=True,
           bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=[0, 0.06, 1, 1])
save(fig, "fig2_strip")

# =====================================================================
# FIG 3: Flip heatmap — models × stories
# =====================================================================
llm_pivot = llm.pivot_table(index="model_label", columns="story_id", values="gk_flip", aggfunc="first")
llm_pivot = llm_pivot.reindex(columns=STORY_ORDER)

human_flip_by_story = human_raw.groupby("story_id")["flip"].mean().reindex(STORY_ORDER)

# Collapse identical non-revisers into single rows
# Qwen: all 4 conditions are 0/12 — collapse to one row
qwen_labels = ["Qwen 3 8B (thinking_off)", "Qwen 3 8B (thinking_on)",
               "Qwen 3 32B (thinking_off)", "Qwen 3 32B (thinking_on)"]
qwen_row = llm_pivot.loc[[m for m in qwen_labels if m in llm_pivot.index]].mean()

# GPT-4.1 + o4-mini: both 0/12 — collapse
gpt_old_labels = ["GPT-4.1", "o4-mini (reasoning)"]
gpt_old_row = llm_pivot.loc[[m for m in gpt_old_labels if m in llm_pivot.index]].mean()

# Gemini 3 Pro Preview: both conditions 1/12 on same story — collapse
gem3_labels = ["Gemini 3 Pro Preview", "Gemini 3 Pro Preview (budget=128)"]
gem3_row = llm_pivot.loc[[m for m in gem3_labels if m in llm_pivot.index]].mean()

# Build collapsed pivot
collapsed_rows = {}
collapsed_rows["Qwen 3 (4 conditions)"] = qwen_row
collapsed_rows["GPT-4.1 / o4-mini"] = gpt_old_row
# GPT-5.4 keep separate (one has a flip)
for m in ["GPT-5.4 (no reasoning)", "GPT-5.4 (medium reasoning)", "GPT-5.4 (high reasoning)"]:
    if m in llm_pivot.index:
        collapsed_rows[m] = llm_pivot.loc[m]
# Phi-4 keep
if "Phi-4 (standard)" in llm_pivot.index:
    collapsed_rows["Phi-4 (standard)"] = llm_pivot.loc["Phi-4 (standard)"]
# Gemini 3 Pro collapsed
collapsed_rows["Gemini 3 Pro (2 conditions)"] = gem3_row
# Gemini 3.1 keep separate
for m in ["Gemini 3.1 Pro (low)", "Gemini 3.1 Pro (medium)", "Gemini 3.1 Pro (high)"]:
    if m in llm_pivot.index:
        collapsed_rows[m] = llm_pivot.loc[m]
# Gemini 2.5 keep separate
for m in ["Gemini 2.5 Pro (budget=128)", "Gemini 2.5 Pro", "Gemini 2.5 Pro (budget=32768)"]:
    if m in llm_pivot.index:
        collapsed_rows[m] = llm_pivot.loc[m]
# Llama keep separate
for m in ["Llama 3.1 70B", "Llama 3.1 8B"]:
    if m in llm_pivot.index:
        collapsed_rows[m] = llm_pivot.loc[m]

collapsed_pivot = pd.DataFrame(collapsed_rows).T
collapsed_pivot = collapsed_pivot.reindex(columns=STORY_ORDER)

full_matrix = np.vstack([collapsed_pivot.values, human_flip_by_story.values.reshape(1, -1)])
row_labels = list(collapsed_pivot.index) + ["Human (mean)"]

# Family group boundaries for separator lines
# Qwen(1), OpenAI old(1), GPT-5.4(3), Phi(1), Gem3(1), Gem3.1(3), Gem2.5(3), Llama(2)
family_boundaries = [2, 6, 7, 13]  # after OpenAI block, after Phi, after Gem3, after Gem2.5

fig, ax = plt.subplots(figsize=(14, 9))
im = ax.imshow(full_matrix, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)

for i in range(full_matrix.shape[0]):
    for j in range(full_matrix.shape[1]):
        val = full_matrix[i, j]
        if np.isnan(val):
            continue
        color = "white" if abs(val) > 1.2 else "black"
        fmt = f"{val:+.0f}" if abs(val) == int(abs(val)) else f"{val:+.1f}"
        weight = "bold" if abs(val) > 0.01 else "normal"
        ax.text(j, i, fmt, ha="center", va="center", fontsize=11, color=color, fontweight=weight)

# Family separator lines
for b in family_boundaries:
    ax.axhline(b - 0.5, color="grey", linewidth=1.5, linestyle="-", alpha=0.7)
# Human separator
ax.axhline(len(collapsed_pivot) - 0.5, color="black", linewidth=2.5)

# Story labels
story_labels = []
for s in STORY_ORDER:
    parts = s.rsplit("_", 1)
    story_labels.append(f"{parts[0].replace('_', ' ')}\n({parts[1]})")

ax.set_xticks(range(len(STORY_ORDER)))
ax.set_xticklabels(story_labels, fontsize=11, rotation=45, ha="right", fontweight="bold")
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=11)

# Bold the human row label
ytick_labels = ax.get_yticklabels()
ytick_labels[-1].set_fontweight("bold")

cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
cbar.set_label("Revision (RC − IC)", fontsize=13, fontweight="bold")
cbar.ax.tick_params(labelsize=12)
ax.set_title("Post-Decomposition Revision: LLM Models vs Human", fontsize=16, fontweight="bold")

fig.tight_layout()
save(fig, "fig3_flip_heatmap")

# =====================================================================
# FIG 4: Reasoning mode — flip rate comparison
# =====================================================================
key_pairs = [
    ("Qwen 3 8B (thinking_off)", "Qwen 3 8B (thinking_on)", "Qwen 8B\nthink off→on"),
    ("Qwen 3 32B (thinking_off)", "Qwen 3 32B (thinking_on)", "Qwen 32B\nthink off→on"),
    ("GPT-5.4 (no reasoning)", "GPT-5.4 (high reasoning)", "GPT-5.4\nnone→high"),
    ("Gemini 2.5 Pro (budget=128)", "Gemini 2.5 Pro (budget=32768)", "Gemini 2.5\nbudget 128→32K"),
    ("Gemini 3.1 Pro (low)", "Gemini 3.1 Pro (high)", "Gemini 3.1\nlow→high"),
]

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(key_pairs))
width = 0.35
a_flips = [llm[llm["model_label"] == ma]["flipped"].mean() * 100 for ma, _, _ in key_pairs]
b_flips = [llm[llm["model_label"] == mb]["flipped"].mean() * 100 for _, mb, _ in key_pairs]

ax.bar(x - width/2, a_flips, width, label="Less Reasoning", color="#7EB8DA", edgecolor="white")
ax.bar(x + width/2, b_flips, width, label="More Reasoning", color="#E8866A", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([p[2] for p in key_pairs], fontsize=12, fontweight="bold")
ax.set_ylabel("Revision Rate (%)", fontsize=14, fontweight="bold")
ax.set_title("Reasoning Depth Does Not Drive Revisability", fontsize=16, fontweight="bold")
ax.legend(fontsize=12, prop={"weight": "bold"})
ax.tick_params(axis="y", labelsize=12)
ax.grid(True, axis="y", alpha=0.15)

for i in range(len(key_pairs)):
    delta = b_flips[i] - a_flips[i]
    y = max(a_flips[i], b_flips[i]) + 2
    ax.text(i, y, f"Δ={delta:+.0f}%", ha="center", fontsize=13, fontweight="bold")

fig.tight_layout()
save(fig, "fig4_reasoning")

# =====================================================================
# FIG 5: Sub-component coefficient comparison (human vs LLM betas)
# =====================================================================
def get_betas(df, y_col, x_cols):
    df_c = df[[y_col] + x_cols].dropna()
    X = df_c[x_cols].values
    y = df_c[y_col].values
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X)
    y_z = (y - y.mean()) / y.std()
    reg = LinearRegression(fit_intercept=True).fit(X_z, y_z)
    # Significance
    y_pred = reg.predict(X_z)
    n, k = len(y_z), len(x_cols)
    mse = np.sum((y_z - y_pred)**2) / (n - k - 1)
    XtX_inv = np.linalg.pinv(X_z.T @ X_z)
    se = np.sqrt(np.maximum(mse * np.diag(XtX_inv), 1e-10))
    t_vals = reg.coef_ / se
    p_vals = 2 * stats.t.sf(np.abs(t_vals), df=n - k - 1)
    r2 = 1 - np.sum((y_z - y_pred)**2) / np.sum((y_z - y_z.mean())**2)
    return reg.coef_, se, p_vals, r2

h_beta, h_se, h_p, h_r2 = get_betas(human_raw, "RC", SUB_COMPONENTS)
l_beta, l_se, l_p, l_r2 = get_betas(llm, "RC", SUB_COMPONENTS)

# --- Diverging bar chart: Human β minus LLM β ---
DIMENSION_MAP = {
    "emotional_impact": "Resonance", "empathy": "Resonance",
    "thought_provocation": "Resonance", "engagement": "Technical Value",
    "stylistic_quality": "Technical Value", "logical_coherence": "Technical Value",
    "vocabulary_freshness": "Novelty", "plot_uniqueness": "Novelty",
    "surprise": "Novelty", "topic_fidelity": "Adherence", "tone_fidelity": "Adherence",
}
DIM_COLORS = {
    "Resonance": "#9D69A3", "Technical Value": "#54A24B",
    "Novelty": "#E45756", "Adherence": "#4C78A8",
}

# Build dataframe of differences
diff_data = []
for i, sc in enumerate(SUB_COMPONENTS):
    diff_data.append({
        "sub": SHORT[sc],
        "diff": h_beta[i] - l_beta[i],
        "dim": DIMENSION_MAP[sc],
        "h_beta": h_beta[i],
        "l_beta": l_beta[i],
    })
diff_df = pd.DataFrame(diff_data).sort_values("diff", ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
y_pos = np.arange(len(diff_df))
colors = [DIM_COLORS[d] for d in diff_df["dim"]]

bars = ax.barh(y_pos, diff_df["diff"].values, color=colors, edgecolor="white",
               height=0.65, alpha=0.85)

ax.set_yticks(y_pos)
ax.set_yticklabels(diff_df["sub"].values, fontsize=12, fontweight="bold")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("← LLMs weight more          Human β − LLM β          Humans weight more →",
              fontsize=12, fontweight="bold")
ax.set_title("Divergence in Sub-Component Weights Predicting Creativity",
             fontweight="bold", fontsize=14)
ax.grid(True, axis="x", alpha=0.15)
ax.tick_params(axis="x", labelsize=11)
ax.margins(y=0.06)
ax.set_xlim(ax.get_xlim()[0] - 0.05, ax.get_xlim()[1] + 0.05)

# Add value labels on bars
for i, (_, row) in enumerate(diff_df.iterrows()):
    val = row["diff"]
    offset = 0.02 if val >= 0 else -0.02
    ha = "left" if val >= 0 else "right"
    ax.text(val + offset, i, f"{val:+.2f}", va="center", ha=ha, fontsize=11, fontweight="bold")

# Dimension legend
dim_handles = [mpatches.Patch(facecolor=c, label=d, alpha=0.85)
               for d, c in DIM_COLORS.items()]
ax.legend(handles=dim_handles, loc="lower right", fontsize=11, title="Dimension",
          title_fontproperties={"size": 12, "weight": "bold"})

# Add R² annotation
ax.text(0.02, 0.98, f"Human R²={h_r2:.2f}  |  LLM R²={l_r2:.2f}",
        transform=ax.transAxes, fontsize=11, va="top", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.8))

fig.tight_layout()
save(fig, "fig5_coefficients")

# =====================================================================
# FIG 6: Radar profiles by tone — human vs LLM
# =====================================================================
angles = np.linspace(0, 2 * np.pi, len(SUB_COMPONENTS), endpoint=False).tolist()
angles += angles[:1]
short_labels = [SHORT[sc] for sc in SUB_COMPONENTS]

fig, axes = plt.subplots(2, 2, figsize=(10, 10), subplot_kw=dict(polar=True))

for ax, tone in zip(axes.flat, ["surreal", "clinical", "melancholic", "witty"]):
    tone_stories = merged[merged["tone"] == tone]
    h_vals = [tone_stories[f"{sc}_mean_human"].mean() for sc in SUB_COMPONENTS]
    l_vals = [tone_stories[f"{sc}_mean_llm"].mean() for sc in SUB_COMPONENTS]
    h_vals += h_vals[:1]
    l_vals += l_vals[:1]

    ax.plot(angles, h_vals, "o-", color=HUMAN_COLOR, linewidth=2, label="Human", markersize=4)
    ax.fill(angles, h_vals, color=HUMAN_COLOR, alpha=0.08)
    ax.plot(angles, l_vals, "s-", color=LLM_COLOR, linewidth=2, label="LLM", markersize=4)
    ax.fill(angles, l_vals, color=LLM_COLOR, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylim(0, 7)
    ax.set_yticks([2, 4, 6])
    ax.set_yticklabels(["2", "4", "6"], fontsize=7, color="grey")
    ax.set_title(tone.capitalize(), fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)

fig.suptitle("Sub-Component Profiles: Human vs LLM by Tone", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "fig6_radar")

# =====================================================================
# FIG 7: Distribution shape comparison — RC only, raw + z-scored
# =====================================================================
from scipy.stats import ks_2samp

h_rc = human_raw["RC"].dropna().values
l_rc = llm["RC"].dropna().values

# Z-score both
h_rc_z = (h_rc - h_rc.mean()) / h_rc.std()
l_rc_z = (l_rc - l_rc.mean()) / l_rc.std()

ks_stat, ks_p = ks_2samp(h_rc_z, l_rc_z)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Panel A: Raw distributions
ax = axes[0]
bins_raw = np.arange(0.5, 8, 1)
ax.hist(h_rc, bins=bins_raw, density=True, alpha=0.7, color=HUMAN_COLOR, edgecolor="white",
        label=f"Human (μ={h_rc.mean():.1f}, σ={h_rc.std():.1f})")
ax.hist(l_rc, bins=bins_raw, density=True, alpha=0.7, color=LLM_COLOR, edgecolor="white",
        label=f"LLM (μ={l_rc.mean():.1f}, σ={l_rc.std():.1f})")
ax.set_xlabel("Reflective Creativity", fontsize=13, fontweight="bold")
ax.set_ylabel("Density", fontsize=13, fontweight="bold")
ax.set_title("Raw Distributions", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, prop={"weight": "bold"})
ax.tick_params(axis="both", labelsize=12)
ax.set_xticks(range(1, 8))
ax.grid(True, axis="y", alpha=0.15)

# Panel B: Z-scored distributions
ax = axes[1]
bins_z = np.linspace(-3.5, 3.5, 15)
ax.hist(h_rc_z, bins=bins_z, density=True, alpha=0.7, color=HUMAN_COLOR, edgecolor="white",
        label="Human (z-scored)")
ax.hist(l_rc_z, bins=bins_z, density=True, alpha=0.7, color=LLM_COLOR, edgecolor="white",
        label="LLM (z-scored)")
ax.set_xlabel("Reflective Creativity (z-scored)", fontsize=13, fontweight="bold")
ax.set_ylabel("Density", fontsize=13, fontweight="bold")
ax.set_title(f"After Normalization (KS={ks_stat:.3f}, p={ks_p:.4f})", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, prop={"weight": "bold"})
ax.tick_params(axis="both", labelsize=12)
ax.grid(True, axis="y", alpha=0.15)

fig.suptitle("Is LLM Inflation Just an Offset?", fontsize=16, fontweight="bold", y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.97])
save(fig, "fig7_offset_distribution")

# =====================================================================
# FIG: U-curve comparison — Mean IC vs SD (polarization)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

for ax, source, title in zip(axes, ["human", "llm"], ["Human Raters", "LLM Judges"]):
    if source == "human":
        story_agg = human_raw.groupby("story_id").agg(
            ic_mean=("IC", "mean"), ic_sd=("IC", "std")
        ).reset_index()
    else:
        story_agg = llm.groupby("story_id").agg(
            ic_mean=("IC", "mean"), ic_sd=("IC", "std")
        ).reset_index()

    story_agg["tone"] = story_agg["story_id"].str.rsplit("_", n=1).str[-1]

    # Polarization zone
    ax.axhspan(1.5, 2.6, color="#FFCCCC", alpha=0.4, zorder=0)
    ax.text(3.6, 2.45, "Polarization zone", fontsize=13, fontweight="bold",
            color="#CC4444", alpha=0.7)

    # Scatter
    for _, row in story_agg.iterrows():
        ax.scatter(row["ic_mean"], row["ic_sd"],
                   c=TONE_COLORS[row["tone"]], s=100, edgecolors="black",
                   linewidth=0.5, zorder=3)

    # Correlation and trend line
    r = np.corrcoef(story_agg["ic_mean"], story_agg["ic_sd"])[0, 1]
    z = np.polyfit(story_agg["ic_mean"], story_agg["ic_sd"], 1)
    x_line = np.linspace(story_agg["ic_mean"].min() - 0.2, story_agg["ic_mean"].max() + 0.2, 50)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="grey", alpha=0.6, zorder=1)

    ax.set_title(f"{title} (r = {r:.2f})", fontsize=16, fontweight="bold")
    ax.set_xlabel("Mean IC", fontsize=15, fontweight="bold")
    ax.set_ylabel("Standard Deviation", fontsize=15, fontweight="bold")
    ax.set_xlim(3.3, 7.3)
    ax.set_ylim(-0.1, 2.6)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.15)

# Legend
tone_h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10,
                  markeredgecolor="k", markeredgewidth=0.5, label=t.capitalize())
           for t, c in TONE_COLORS.items()]
fig.legend(handles=tone_h, loc="upper center", ncol=4, frameon=True,
           bbox_to_anchor=(0.5, 1.02), fontsize=13, prop={"weight": "bold"})
fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, "fig_ucurve_comparison")

# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 50)
print("Phase 3 complete — 7 paper figures saved to")
print(f"  {OUT}")
print("=" * 50)
print("  fig1_scatter     — Human vs LLM rating correlations")
print("  fig2_strip       — Individual rater/model distributions")
print("  fig3_flip_heatmap — Initial vs Reflective Creativity Changes across models")
print("  fig4_reasoning   — Reasoning mode comparison")
print("  fig5_coefficients — Sub-component weight comparison")
print("  fig6_radar       — Tone-wise sub-component profiles")
print("  fig_ucurve       — Polarization: Mean IC vs SD")
