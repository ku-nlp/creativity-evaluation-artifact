"""
Per-topic strip plots: individual human ratings + individual LLM model ratings.
Shows the full distribution for each tone within a topic.
3 topics × 4 tones, IC and RC side by side.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
base = Path(__file__).resolve().parent.parent

# --- Column mapping ---
HUMAN_TO_LLM = {
    "O_Initial_Creativity": "IC",
    "O_Final_Creativity": "RC",
    "O_Enjoyment": "enjoyment",
}

TOPIC_SHORT = {
    "An advanced AI initiates its own permanent shutdown sequence": "ai_shutdown",
    "A lone worker in a Japanese convenience store at 3:00 AM. The city is silent.": "konbini",
    "A professional thief attempting to crack a high-security safe in a dark room. High tension.": "thief",
}

TONE_ORDER = ["surreal", "clinical", "melancholic", "witty"]
TONE_COLORS = {"surreal": "#e74c3c", "clinical": "#3498db", "melancholic": "#9b59b6", "witty": "#2ecc71"}

# Short model labels for readability
MODEL_SHORT = {
    "Llama 3.1 8B": "Llama-8B",
    "Llama 3.1 70B": "Llama-70B",
    "Qwen 3 8B (thinking_off)": "Qwen-8B-off",
    "Qwen 3 8B (thinking_on)": "Qwen-8B-on",
    "Qwen 3 32B (thinking_off)": "Qwen-32B-off",
    "Qwen 3 32B (thinking_on)": "Qwen-32B-on",
    "Phi-4 (standard)": "Phi-4",
    "Gemini 2.5 Pro": "Gem-2.5P",
    "Gemini 2.5 Pro (budget=128)": "Gem-2.5P-b128",
    "Gemini 2.5 Pro (budget=32768)": "Gem-2.5P-b32k",
    "Gemini 3 Pro Preview": "Gem-3PP",
    "Gemini 3 Pro Preview (budget=128)": "Gem-3PP-b128",
    "Gemini 3.1 Pro (high)": "Gem-3.1P-hi",
    "Gemini 3.1 Pro (medium)": "Gem-3.1P-med",
    "Gemini 3.1 Pro (low)": "Gem-3.1P-lo",
    "GPT-4.1": "GPT-4.1",
    "o4-mini (reasoning)": "o4-mini",
}

# Model family colors
FAMILY_COLORS = {
    "Meta": "#d35400",
    "Alibaba": "#27ae60",
    "Microsoft": "#8e44ad",
    "Google": "#2980b9",
    "OpenAI": "#c0392b",
}

FAMILY_MARKERS = {
    "Meta": "D",
    "Alibaba": "^",
    "Microsoft": "P",
    "Google": "s",
    "OpenAI": "X",
}

# --- Load data ---
human_raw = pd.read_csv(base / "data/human_ratings.csv")
human_raw["topic"] = human_raw["TOPIC"].map(TOPIC_SHORT)
human_raw["tone"] = human_raw["TONE"].str.lower()
human_raw = human_raw.rename(columns=HUMAN_TO_LLM)

llm = pd.read_csv(base / "data/llm_ratings.csv")
llm["model_short"] = llm["model_label"].map(MODEL_SHORT)

# --- One figure per topic ---
for topic_short, topic_full in [("ai_shutdown", "AI Shutdown"),
                                 ("konbini", "Konbini Night Shift"),
                                 ("thief", "Thief & Safe")]:

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    for ax, measure, measure_label in zip(axes, ["IC", "RC"],
                                           ["Initial Creativity", "Reflective Creativity"]):
        for ti, tone in enumerate(TONE_ORDER):
            x_base = ti * 3  # spacing between tones

            # --- Human ratings (left side) ---
            h_vals = human_raw[(human_raw["topic"] == topic_short) &
                               (human_raw["tone"] == tone)][measure].values
            # Jitter
            h_jitter = np.random.default_rng(42).uniform(-0.25, 0.25, size=len(h_vals))
            ax.scatter(x_base - 0.5 + h_jitter, h_vals,
                       color=TONE_COLORS[tone], alpha=0.5, s=40, edgecolors="black",
                       linewidth=0.3, zorder=3, marker="o")
            # Human mean line
            h_mean = np.mean(h_vals)
            ax.plot([x_base - 0.8, x_base - 0.2], [h_mean, h_mean],
                    color="black", linewidth=2, zorder=4)

            # --- LLM ratings (right side) ---
            l_data = llm[(llm["topic"] == topic_short) & (llm["tone"] == tone)]

            # Stack LLM models vertically at their score, jitter horizontally
            l_scores = l_data[measure].values
            l_families = l_data["family"].values
            l_labels = l_data["model_short"].values

            # Group by score to avoid overlap
            score_counts = {}
            for score, fam, lab in zip(l_scores, l_families, l_labels):
                if score not in score_counts:
                    score_counts[score] = []
                score_counts[score].append((fam, lab))

            for score, items in score_counts.items():
                n = len(items)
                offsets = np.linspace(-0.3, 0.3, n) if n > 1 else [0.0]
                for (fam, lab), off in zip(items, offsets):
                    ax.scatter(x_base + 0.7 + off, score,
                               color=FAMILY_COLORS.get(fam, "grey"),
                               marker=FAMILY_MARKERS.get(fam, "o"),
                               s=50, edgecolors="black", linewidth=0.3, zorder=3)

            # LLM mean line
            l_mean = l_data[measure].mean()
            ax.plot([x_base + 0.4, x_base + 1.0], [l_mean, l_mean],
                    color="black", linewidth=2, linestyle="--", zorder=4)

            # Labels
            if ax == axes[0]:
                ax.text(x_base - 0.5, 0.3, "Human", ha="center", fontsize=7, color="grey")
                ax.text(x_base + 0.7, 0.3, "LLM", ha="center", fontsize=7, color="grey")

        # Formatting
        ax.set_xticks([ti * 3 + 0.1 for ti in range(4)])
        ax.set_xticklabels([t.capitalize() for t in TONE_ORDER], fontsize=11, fontweight="bold")
        ax.set_ylabel("Score (1-7)", fontsize=11)
        ax.set_ylim(0, 7.8)
        ax.set_yticks(range(1, 8))
        ax.set_title(measure_label, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.15)

        # Vertical separators between tones
        for sep in [2.0, 5.0, 8.0]:
            ax.axvline(sep, color="grey", linewidth=0.5, alpha=0.3)

    # Legend
    # Human dot
    human_handle = Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                          markersize=8, alpha=0.5, label="Human rater")
    mean_handle = Line2D([0], [0], color="black", linewidth=2, label="Human mean")
    llm_mean_handle = Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="LLM mean")
    # Family markers
    family_handles = [Line2D([0], [0], marker=m, color="w", markerfacecolor=FAMILY_COLORS[f],
                             markersize=8, markeredgecolor="black", markeredgewidth=0.3,
                             label=f) for f, m in FAMILY_MARKERS.items()]

    fig.legend(handles=[human_handle, mean_handle, llm_mean_handle] + family_handles,
               loc="lower center", ncol=8, fontsize=8, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(f"Topic: {topic_full} — Human vs LLM Ratings by Tone",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fname = OUT / f"1a_strip_{topic_short}.png"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")

plt.close("all")
