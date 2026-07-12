"""
Phase B: Human Data Reliability

B1: ICC(2,k) for each measure
B2: Split-half reliability (1000 iterations)
B3: Bootstrap CIs on human story means

Output: new_figures/phaseB_*.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg
from scipy import stats
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FIG_DIR = BASE / "analysis/figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Column groups ────────────────────────────────────────────────────────────
HOLISTIC = ["O_Enjoyment", "O_Initial_Creativity", "O_Final_Creativity"]
SUB_COLS = [
    "R_Emotion", "A_Topic", "N_Vocab", "N_Plot", "N_Surprise",
    "R_Empathy", "R_Thought", "V_Engagement", "V_Style", "V_Logic", "A_Tone",
]
ALL_MEASURES = HOLISTIC + SUB_COLS

SHORT_NAMES = {
    "O_Enjoyment": "enjoyment",
    "O_Initial_Creativity": "IC",
    "O_Final_Creativity": "RC",
    "R_Emotion": "emotion",
    "A_Topic": "topic",
    "N_Vocab": "vocab",
    "N_Plot": "plot",
    "N_Surprise": "surprise",
    "R_Empathy": "empathy",
    "R_Thought": "thought",
    "V_Engagement": "engage",
    "V_Style": "style",
    "V_Logic": "logic",
    "A_Tone": "tone",
}

TOPIC_MAP = {
    "An advanced AI initiates its own permanent shutdown sequence": "ai_shutdown",
    "A lone worker in a Japanese convenience store at 3:00 AM. The city is silent.": "konbini",
    "A professional thief attempting to crack a high-security safe in a dark room. High tension.": "thief",
}


def load_data():
    human = pd.read_csv(BASE / "data/human_ratings.csv")
    human["topic"] = human["TOPIC"].map(TOPIC_MAP)
    human["tone"] = human["TONE"].str.lower()
    human["story_id"] = human["topic"] + "_" + human["tone"]
    return human


def compute_icc(human, measure):
    """
    Compute ICC(2,k) for a measure across stories.
    Handles unbalanced design by trimming to min raters per story.
    """
    # Build long-format with balanced design
    min_raters = human.groupby("story_id").size().min()
    rows = []
    for story_id, group in human.groupby("story_id"):
        for rater_idx, (_, row) in enumerate(group.head(min_raters).iterrows()):
            rows.append({
                "story": story_id,
                "rater": rater_idx,
                "score": row[measure],
            })
    df = pd.DataFrame(rows)

    icc_result = pg.intraclass_corr(
        data=df, targets="story", raters="rater", ratings="score"
    )
    # ICC(A,k) = two-way random, average measures (equivalent to ICC2k)
    icc_row = icc_result[icc_result["Type"] == "ICC(A,k)"]
    if len(icc_row) == 0:
        return None, None, None
    row = icc_row.iloc[0]
    ci = row["CI95"]
    return row["ICC"], ci[0], ci[1]


def split_half_reliability(human, measure, n_iter=1000, seed=42):
    """
    Split raters randomly into two halves, compute story means for each half,
    correlate. Repeat n_iter times.
    """
    rng = np.random.RandomState(seed)
    correlations = []
    stories = human["story_id"].unique()

    for _ in range(n_iter):
        means_a, means_b = [], []
        for story in stories:
            group = human[human["story_id"] == story][measure].values
            n = len(group)
            if n < 2:
                continue
            perm = rng.permutation(n)
            half = n // 2
            means_a.append(group[perm[:half]].mean())
            means_b.append(group[perm[half:half * 2]].mean())
        r, _ = stats.spearmanr(means_a, means_b)
        correlations.append(r)
    return np.array(correlations)


def bootstrap_story_means(human, measure, n_boot=10000, seed=42):
    """
    For each story, bootstrap resample raters and compute mean.
    Returns dict: story_id → (mean, ci_low, ci_high).
    """
    rng = np.random.RandomState(seed)
    results = {}
    for story_id, group in human.groupby("story_id"):
        scores = group[measure].values
        n = len(scores)
        boot_means = np.array([
            rng.choice(scores, size=n, replace=True).mean()
            for _ in range(n_boot)
        ])
        ci = np.percentile(boot_means, [2.5, 97.5])
        results[story_id] = (scores.mean(), ci[0], ci[1])
    return results


def plot_icc_bars(icc_data):
    """Bar chart of ICC values with CI error bars."""
    measures = list(icc_data.keys())
    iccs = [icc_data[m][0] for m in measures]
    ci_lo = [icc_data[m][1] for m in measures]
    ci_hi = [icc_data[m][2] for m in measures]
    labels = [SHORT_NAMES.get(m, m) for m in measures]

    yerr_lo = [max(0, v - lo) for v, lo in zip(iccs, ci_lo)]
    yerr_hi = [hi - v for v, hi in zip(iccs, ci_hi)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(measures))
    colors = ["tab:green" if v >= 0.6 else "tab:orange" if v >= 0.4 else "tab:red"
              for v in iccs]
    ax.bar(x, iccs, color=colors, alpha=0.7)
    ax.errorbar(x, iccs, yerr=[yerr_lo, yerr_hi], fmt="none", color="black", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("ICC(2,k)")
    ax.set_title("Human Inter-Rater Reliability (ICC2k, two-way random, average)")
    ax.axhline(0.6, ls="--", color="gray", alpha=0.5, label="acceptable (0.6)")
    ax.axhline(0.4, ls="--", color="red", alpha=0.3, label="poor (0.4)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phaseB_icc.png", dpi=200)
    plt.close()


def plot_bootstrap_ci(boot_data, measure_label):
    """Forest plot of bootstrap CIs per story."""
    stories = sorted(boot_data.keys())
    means = [boot_data[s][0] for s in stories]
    ci_lo = [boot_data[s][1] for s in stories]
    ci_hi = [boot_data[s][2] for s in stories]

    fig, ax = plt.subplots(figsize=(8, 6))
    y = range(len(stories))
    ax.errorbar(means, y, xerr=[[m - lo for m, lo in zip(means, ci_lo)],
                                 [hi - m for m, hi in zip(means, ci_hi)]],
                fmt="o", color="tab:blue", capsize=4, markersize=5)
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("_", " ") for s in stories], fontsize=8)
    ax.set_xlabel(f"Mean {measure_label} (95% Bootstrap CI)")
    ax.set_title(f"Human Story Means — {measure_label}")
    ax.axvline(4, ls="--", color="gray", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"phaseB_bootstrap_{measure_label}.png", dpi=200)
    plt.close()


def main():
    human = load_data()
    print(f"Human data: {len(human)} rows, {human['story_id'].nunique()} stories\n")

    # ── B1: ICC ───────────────────────────────────────────────────────────
    print("=== B1: ICC(2,k) ===")
    icc_data = {}
    print(f"  {'Measure':<25} {'ICC':>6} {'95% CI':>16} {'Quality'}")
    print(f"  {'-'*60}")
    for m in ALL_MEASURES:
        try:
            icc, lo, hi = compute_icc(human, m)
            icc_data[m] = (icc, lo, hi)
            quality = "good" if icc >= 0.6 else "moderate" if icc >= 0.4 else "poor"
            print(f"  {SHORT_NAMES.get(m, m):<25} {icc:>6.3f} [{lo:.3f}, {hi:.3f}]  {quality}")
        except Exception as e:
            print(f"  {SHORT_NAMES.get(m, m):<25}  ERROR: {e}")

    # ── B2: Split-half reliability ────────────────────────────────────────
    print("\n=== B2: Split-Half Reliability (1000 iterations) ===")
    print(f"  {'Measure':<25} {'Mean r':>8} {'95% CI':>18}")
    print(f"  {'-'*55}")
    for m in ALL_MEASURES:
        corrs = split_half_reliability(human, m)
        ci = np.percentile(corrs, [2.5, 97.5])
        print(f"  {SHORT_NAMES.get(m, m):<25} {corrs.mean():>8.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

    # ── B3: Bootstrap CIs ─────────────────────────────────────────────────
    print("\n=== B3: Bootstrap CIs on Story Means (IC) ===")
    boot_ic = bootstrap_story_means(human, "O_Initial_Creativity")
    print(f"  {'Story':<30} {'Mean':>6} {'95% CI':>16} {'Width':>6}")
    print(f"  {'-'*62}")
    for story in sorted(boot_ic.keys()):
        mean, lo, hi = boot_ic[story]
        print(f"  {story:<30} {mean:>6.2f} [{lo:.2f}, {hi:.2f}] {hi - lo:>6.2f}")

    boot_rc = bootstrap_story_means(human, "O_Final_Creativity")

    # ── Plots ─────────────────────────────────────────────────────────────
    if icc_data:
        plot_icc_bars(icc_data)
    plot_bootstrap_ci(boot_ic, "IC")
    plot_bootstrap_ci(boot_rc, "RC")

    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
