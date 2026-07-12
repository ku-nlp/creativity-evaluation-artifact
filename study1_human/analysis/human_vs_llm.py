"""
Phase C: Individual Condition vs Human Distribution

C1: Per-condition human-likeness profile (rank corr, bias, percentile)
C2: Per-story violin + LLM overlay plots
C3: Reasoning gradients — does the knob move scores toward humans?
C4: Sub-component bias heatmap (conditions × 11 components)

Output: new_figures/phaseC_*.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FIG_DIR = BASE / "analysis/figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Column mappings ──────────────────────────────────────────────────────────
HUMAN_SUB_MAP = {
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
LLM_SUB_COLS = list(HUMAN_SUB_MAP.values())
SHORT_LABELS = {
    "emotional_impact": "emotion", "topic_fidelity": "topic",
    "vocabulary_freshness": "vocab", "plot_uniqueness": "plot",
    "surprise": "surprise", "empathy": "empathy",
    "thought_provocation": "thought", "engagement": "engage",
    "stylistic_quality": "style", "logical_coherence": "logic",
    "tone_fidelity": "tone",
}

TOPIC_MAP = {
    "An advanced AI initiates its own permanent shutdown sequence": "ai_shutdown",
    "A lone worker in a Japanese convenience store at 3:00 AM. The city is silent.": "konbini",
    "A professional thief attempting to crack a high-security safe in a dark room. High tension.": "thief",
}

# GPT-5.4 labels (excluded from human comparison C1/C2, included in C3)
GPT54_LABELS = [
    "GPT-5.4 (no reasoning)",
    "GPT-5.4 (medium reasoning)",
    "GPT-5.4 (high reasoning)",
]

# Reasoning gradient families
REASONING_GRADIENTS = {
    "Gemini 3.1 Pro\n(thinking level)": {
        "labels": ["Gemini 3.1 Pro (low)", "Gemini 3.1 Pro (medium)", "Gemini 3.1 Pro (high)"],
        "x_labels": ["low", "medium", "high"],
    },
    "Gemini 2.5 Pro\n(thinking budget)": {
        "labels": ["Gemini 2.5 Pro (budget=128)", "Gemini 2.5 Pro", "Gemini 2.5 Pro (budget=32768)"],
        "x_labels": ["128", "default", "32768"],
    },
    "Qwen 3 8B\n(thinking)": {
        "labels": ["Qwen 3 8B (thinking_off)", "Qwen 3 8B (thinking_on)"],
        "x_labels": ["off", "on"],
    },
    "Qwen 3 32B\n(thinking)": {
        "labels": ["Qwen 3 32B (thinking_off)", "Qwen 3 32B (thinking_on)"],
        "x_labels": ["off", "on"],
    },
    "OpenAI\n(model)": {
        "labels": ["GPT-4.1", "o4-mini (reasoning)"],
        "x_labels": ["GPT-4.1", "o4-mini"],
    },
    "GPT-5.4\n(reasoning effort)": {
        "labels": ["GPT-5.4 (no reasoning)", "GPT-5.4 (medium reasoning)", "GPT-5.4 (high reasoning)"],
        "x_labels": ["none", "medium", "high"],
    },
}

FAMILY_COLORS = {
    "Meta": "tab:red",
    "Alibaba": "tab:green",
    "Microsoft": "tab:purple",
    "Google": "tab:blue",
    "OpenAI": "tab:orange",
}

FAMILY_MARKERS = {
    "Meta": "D",
    "Alibaba": "^",
    "Microsoft": "s",
    "Google": "o",
    "OpenAI": "P",
}


def load_data():
    human = pd.read_csv(BASE / "data/human_ratings.csv")
    llm = pd.read_csv(BASE / "data/llm_ratings.csv")

    human["topic"] = human["TOPIC"].map(TOPIC_MAP)
    human["tone"] = human["TONE"].str.lower()
    human["story_id"] = human["topic"] + "_" + human["tone"]

    # Rename human sub-component columns to match LLM
    human = human.rename(columns=HUMAN_SUB_MAP)
    human = human.rename(columns={
        "O_Initial_Creativity": "IC",
        "O_Final_Creativity": "RC",
        "O_Enjoyment": "enjoyment",
    })
    return human, llm


def compute_human_story_means(human):
    """Compute mean per story for all measures."""
    measures = ["IC", "RC", "enjoyment"] + LLM_SUB_COLS
    return human.groupby("story_id")[measures].mean()


def c1_human_likeness(human, llm, human_means):
    """Per-condition human-likeness scorecard."""
    stories = sorted(human_means.index)
    # Exclude GPT-5.4 from human comparison
    llm_no54 = llm[~llm["model_label"].isin(GPT54_LABELS)]
    conditions = sorted(llm_no54["model_label"].unique())

    results = []
    for cond in conditions:
        cond_data = llm_no54[llm_no54["model_label"] == cond].set_index("story_id")
        family = cond_data["family"].iloc[0]

        for measure in ["IC", "RC", "enjoyment"]:
            h_vals = human_means.loc[stories, measure].values
            l_vals = cond_data.loc[stories, measure].values
            rho, p = stats.spearmanr(h_vals, l_vals)
            bias = (l_vals - h_vals).mean()

            # Percentile: for each story, what % of human raters scored <= LLM score
            pct_ranks = []
            for story in stories:
                h_scores = human[human["story_id"] == story][measure].values
                l_score = cond_data.loc[story, measure]
                pct = (h_scores <= l_score).mean() * 100
                pct_ranks.append(pct)

            results.append({
                "condition": cond,
                "family": family,
                "measure": measure,
                "spearman_r": rho,
                "spearman_p": p,
                "mean_bias": bias,
                "median_percentile": np.median(pct_ranks),
            })

    df = pd.DataFrame(results)
    return df


def c2_story_violins(human, llm, measure="IC"):
    """Violin plots of human distributions + LLM points overlaid."""
    stories = sorted(human["story_id"].unique())
    llm_no54 = llm[~llm["model_label"].isin(GPT54_LABELS)]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharey=True)
    axes = axes.flatten()

    for idx, story in enumerate(stories):
        ax = axes[idx]
        h_scores = human[human["story_id"] == story][measure].values

        # Human violin
        parts = ax.violinplot([h_scores], positions=[0], showmeans=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("lightblue")
            pc.set_alpha(0.6)

        # LLM points
        llm_story = llm_no54[llm_no54["story_id"] == story]
        for _, row in llm_story.iterrows():
            jitter = np.random.uniform(-0.15, 0.15)
            ax.scatter(
                0.4 + jitter, row[measure],
                color=FAMILY_COLORS.get(row["family"], "gray"),
                marker=FAMILY_MARKERS.get(row["family"], "o"),
                s=30, alpha=0.8, edgecolors="black", linewidths=0.3,
            )

        ax.set_title(story.replace("_", "\n"), fontsize=8)
        ax.set_xticks([0, 0.4])
        ax.set_xticklabels(["Human", "LLM"], fontsize=7)
        ax.set_ylim(0.5, 7.5)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker=FAMILY_MARKERS[f], color="w",
               markerfacecolor=FAMILY_COLORS[f], markersize=8, label=f)
        for f in FAMILY_COLORS
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=9)
    fig.suptitle(f"Human Distribution vs LLM Scores — {measure}", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(FIG_DIR / f"phaseC_violins_{measure}.png", dpi=200)
    plt.close()


def c3_reasoning_gradients(llm, human_means):
    """Plot how human-likeness changes across reasoning levels within each family."""
    stories = sorted(human_means.index)
    n_families = len(REASONING_GRADIENTS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax_idx, (family_name, cfg) in enumerate(REASONING_GRADIENTS.items()):
        ax = axes[ax_idx]
        labels = cfg["labels"]
        x_labels = cfg["x_labels"]

        # For each condition in gradient, compute Spearman r with human means
        for measure, color, marker in [
            ("IC", "tab:blue", "o"),
            ("RC", "tab:red", "s"),
            ("enjoyment", "tab:green", "^"),
        ]:
            rhos = []
            for cond_label in labels:
                cond_data = llm[llm["model_label"] == cond_label].set_index("story_id")
                if len(cond_data) == 0:
                    rhos.append(np.nan)
                    continue
                h_vals = human_means.loc[stories, measure].values
                l_vals = cond_data.loc[stories, measure].values
                rho, _ = stats.spearmanr(h_vals, l_vals)
                rhos.append(rho)

            ax.plot(range(len(x_labels)), rhos, f"-{marker}", color=color,
                    label=measure, markersize=6)

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel("Spearman r with humans")
        ax.set_title(family_name, fontsize=10)
        ax.set_ylim(-0.2, 1.0)
        ax.axhline(0, ls="--", color="gray", alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=8)

    plt.suptitle("Reasoning Gradients: Does More Thinking = More Human-Like?", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(FIG_DIR / "phaseC_reasoning_gradients.png", dpi=200)
    plt.close()


def c4_bias_heatmap(llm, human_means):
    """Heatmap: conditions × sub-components, colored by bias from human mean."""
    stories = sorted(human_means.index)
    llm_no54 = llm[~llm["model_label"].isin(GPT54_LABELS)]
    conditions = sorted(llm_no54["model_label"].unique())

    bias_matrix = []
    for cond in conditions:
        cond_data = llm_no54[llm_no54["model_label"] == cond].set_index("story_id")
        biases = []
        for sub in LLM_SUB_COLS:
            h_vals = human_means.loc[stories, sub].values
            l_vals = cond_data.loc[stories, sub].values
            biases.append((l_vals - h_vals).mean())
        bias_matrix.append(biases)

    bias_df = pd.DataFrame(
        bias_matrix,
        index=conditions,
        columns=[SHORT_LABELS[s] for s in LLM_SUB_COLS],
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(bias_df.values, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(len(bias_df.columns)))
    ax.set_xticklabels(bias_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(bias_df.index)))
    ax.set_yticklabels(bias_df.index, fontsize=8)
    ax.set_title("Sub-Component Bias: LLM − Human Mean")

    # Annotate cells
    for i in range(len(bias_df.index)):
        for j in range(len(bias_df.columns)):
            val = bias_df.values[i, j]
            color = "white" if abs(val) > 1.5 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Bias (LLM − Human)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phaseC_bias_heatmap.png", dpi=200)
    plt.close()
    return bias_df


def main():
    human, llm = load_data()
    human_means = compute_human_story_means(human)
    print(f"Human: {len(human)} rows, LLM: {len(llm)} rows")
    print(f"LLM conditions: {llm['model_label'].nunique()}")

    # ── C1: Human-likeness scorecard ──────────────────────────────────────
    scorecard = c1_human_likeness(human, llm, human_means)

    print("\n=== C1: Human-Likeness Scorecard (IC) ===")
    ic_scores = scorecard[scorecard["measure"] == "IC"].sort_values("spearman_r", ascending=False)
    print(f"  {'Condition':<35} {'Family':<10} {'Spearman r':>10} {'p':>8} {'Bias':>6} {'Pctl':>6}")
    print(f"  {'-'*80}")
    for _, row in ic_scores.iterrows():
        sig = "*" if row["spearman_p"] < 0.05 else ""
        print(f"  {row['condition']:<35} {row['family']:<10} {row['spearman_r']:>9.3f}{sig} {row['spearman_p']:>7.3f} {row['mean_bias']:>+5.2f} {row['median_percentile']:>5.1f}")

    print("\n=== C1: Human-Likeness Scorecard (RC) ===")
    rc_scores = scorecard[scorecard["measure"] == "RC"].sort_values("spearman_r", ascending=False)
    print(f"  {'Condition':<35} {'Family':<10} {'Spearman r':>10} {'p':>8} {'Bias':>6} {'Pctl':>6}")
    print(f"  {'-'*80}")
    for _, row in rc_scores.iterrows():
        sig = "*" if row["spearman_p"] < 0.05 else ""
        print(f"  {row['condition']:<35} {row['family']:<10} {row['spearman_r']:>9.3f}{sig} {row['spearman_p']:>7.3f} {row['mean_bias']:>+5.2f} {row['median_percentile']:>5.1f}")

    # ── C2: Violin plots ─────────────────────────────────────────────────
    c2_story_violins(human, llm, "IC")
    c2_story_violins(human, llm, "RC")

    # ── C3: Reasoning gradients ──────────────────────────────────────────
    c3_reasoning_gradients(llm, human_means)

    # ── C4: Bias heatmap ─────────────────────────────────────────────────
    bias_df = c4_bias_heatmap(llm, human_means)
    print("\n=== C4: Mean Bias by Sub-Component (across all conditions) ===")
    mean_bias = bias_df.mean()
    for comp, val in mean_bias.sort_values(ascending=False).items():
        print(f"  {comp:<12} {val:>+5.2f}")

    # Save scorecard
    scorecard.to_csv(FIG_DIR / "phaseC_scorecard.csv", index=False)
    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
