"""
Phase A: Framework Validation — PCA on Human + LLM sub-component ratings.

A1: PCA on human ratings (N=115)
A2: PCA on LLM ratings (1 representative per family, N=60)
A3: Tucker's congruence coefficient between factor loadings
A4: Ceiling effect analysis (all 11 vs 9 sub-components)

Output: new_figures/phaseA_*.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FIG_DIR = BASE / "analysis/figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Column mappings ──────────────────────────────────────────────────────────
HUMAN_SUB_COLS = [
    "R_Emotion", "A_Topic", "N_Vocab", "N_Plot", "N_Surprise",
    "R_Empathy", "R_Thought", "V_Engagement", "V_Style", "V_Logic", "A_Tone",
]
LLM_SUB_COLS = [
    "emotional_impact", "topic_fidelity", "vocabulary_freshness",
    "plot_uniqueness", "surprise", "empathy", "thought_provocation",
    "engagement", "stylistic_quality", "logical_coherence", "tone_fidelity",
]
# Shared short labels (same order as both lists above)
SHORT_LABELS = [
    "emotion", "topic", "vocab", "plot", "surprise",
    "empathy", "thought", "engage", "style", "logic", "tone",
]
# Indices of fidelity pair (topic=1, tone=10) for the 9-component variant
FIDELITY_IDX = [1, 10]
SHORT_LABELS_9 = [s for i, s in enumerate(SHORT_LABELS) if i not in FIDELITY_IDX]

# ── Representatives: 1 baseline per family ───────────────────────────────────
REPRESENTATIVES = {
    "Meta":      "Llama 3.1 8B",
    "Alibaba":   "Qwen 3 8B (thinking_off)",
    "Microsoft": "Phi-4 (standard)",
    "Google":    "Gemini 2.5 Pro",
    "OpenAI":    "GPT-4.1",
}

# ── Topic mapping (human full text → short key) ─────────────────────────────
TOPIC_MAP = {
    "An advanced AI initiates its own permanent shutdown sequence": "ai_shutdown",
    "A lone worker in a Japanese convenience store at 3:00 AM. The city is silent.": "konbini",
    "A professional thief attempting to crack a high-security safe in a dark room. High tension.": "thief",
}


def load_data():
    human = pd.read_csv(BASE / "data/human_ratings.csv")
    llm = pd.read_csv(BASE / "data/llm_ratings.csv")

    # Map human topics to short keys
    human["topic"] = human["TOPIC"].map(TOPIC_MAP)
    human["tone"] = human["TONE"].str.lower()
    human["story_id"] = human["topic"] + "_" + human["tone"]

    # Filter LLM to representatives
    rep_labels = list(REPRESENTATIVES.values())
    llm_rep = llm[llm["model_label"].isin(rep_labels)].copy()
    return human, llm, llm_rep


def run_pca(X, n_components=None):
    """Standardize and run PCA. Returns (pca_object, scaled_data, scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if n_components is None:
        n_components = X_scaled.shape[1]
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    return pca, X_scaled


def tucker_congruence(A, B):
    """Tucker's congruence coefficient between corresponding columns of A and B."""
    # A, B are (n_features, n_components) loading matrices
    coeffs = []
    for j in range(A.shape[1]):
        a, b = A[:, j], B[:, j]
        rc = np.dot(a, b) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b)))
        coeffs.append(rc)
    return np.array(coeffs)


def sign_align(loadings_ref, loadings_target):
    """Flip signs of target loadings to match reference (maximize congruence)."""
    aligned = loadings_target.copy()
    for j in range(aligned.shape[1]):
        if np.dot(loadings_ref[:, j], aligned[:, j]) < 0:
            aligned[:, j] *= -1
    return aligned


def plot_scree(pca_human, pca_llm, suffix=""):
    """Side-by-side scree plots."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, pca, title in zip(axes, [pca_human, pca_llm], ["Human", "LLM"]):
        evr = pca.explained_variance_ratio_
        cumvar = np.cumsum(evr)
        n = len(evr)
        ax.bar(range(1, n + 1), evr, alpha=0.6, label="Individual")
        ax.plot(range(1, n + 1), cumvar, "o-", color="tab:red", label="Cumulative")
        ax.set_xlabel("Component")
        ax.set_ylabel("Variance Explained")
        ax.set_title(f"{title} (N={pca.n_features_in_}d)")
        ax.set_xticks(range(1, n + 1))
        ax.legend(fontsize=8)
        ax.axhline(0.1, ls="--", color="gray", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"phaseA_scree{suffix}.png", dpi=200)
    plt.close()


def plot_loadings(loadings_h, loadings_l, labels, congruence, suffix=""):
    """Side-by-side loading bar charts for first 2 components."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, comp in enumerate([0, 1]):
        for col, (loadings, source) in enumerate(
            [(loadings_h, "Human"), (loadings_l, "LLM")]
        ):
            ax = axes[row, col]
            vals = loadings[:, comp]
            colors = ["tab:blue" if v >= 0 else "tab:red" for v in vals]
            ax.barh(range(len(labels)), vals, color=colors, alpha=0.7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_title(
                f"{source} — PC{comp + 1} "
                f"({congruence[comp]:+.2f} congruence)"
                if col == 1
                else f"{source} — PC{comp + 1}"
            )
            ax.axvline(0, color="black", lw=0.5)
            ax.set_xlim(-1, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"phaseA_loadings{suffix}.png", dpi=200)
    plt.close()


def plot_ceiling(human_pct, llm_pct, labels):
    """Bar chart: % at max (7) for each sub-component, human vs LLM."""
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, human_pct, w, label="Human", color="tab:blue", alpha=0.7)
    ax.bar(x + w / 2, llm_pct, w, label="LLM (reps)", color="tab:orange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% ratings at maximum (7)")
    ax.set_title("Ceiling Effect: Human vs LLM")
    ax.legend()
    ax.axhline(50, ls="--", color="gray", alpha=0.4, label="50%")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phaseA_ceiling.png", dpi=200)
    plt.close()


def main():
    human, llm_all, llm_rep = load_data()
    print(f"Human: {len(human)} rows, LLM reps: {len(llm_rep)} rows")
    print(f"LLM reps: {llm_rep['model_label'].unique()}")

    # ── A1 & A2: PCA (all 11 sub-components) ─────────────────────────────
    X_human = human[HUMAN_SUB_COLS].values.astype(float)
    X_llm = llm_rep[LLM_SUB_COLS].values.astype(float)

    pca_h, _ = run_pca(X_human)
    pca_l, _ = run_pca(X_llm)

    print("\n=== A1: Human PCA ===")
    for i, (ev, evr) in enumerate(
        zip(pca_h.explained_variance_, pca_h.explained_variance_ratio_)
    ):
        print(f"  PC{i+1}: eigenvalue={ev:.2f}, variance={evr:.1%}, cumulative={sum(pca_h.explained_variance_ratio_[:i+1]):.1%}")

    print("\n=== A2: LLM PCA ===")
    for i, (ev, evr) in enumerate(
        zip(pca_l.explained_variance_, pca_l.explained_variance_ratio_)
    ):
        print(f"  PC{i+1}: eigenvalue={ev:.2f}, variance={evr:.1%}, cumulative={sum(pca_l.explained_variance_ratio_[:i+1]):.1%}")

    # ── A3: Factor congruence ─────────────────────────────────────────────
    loadings_h = pca_h.components_.T  # (11, n_components)
    loadings_l = pca_l.components_.T
    loadings_l_aligned = sign_align(loadings_h, loadings_l)
    congruence = tucker_congruence(loadings_h, loadings_l_aligned)

    print("\n=== A3: Tucker's Congruence (11 components) ===")
    for i, rc in enumerate(congruence[:4]):
        print(f"  PC{i+1}: rc={rc:.3f}  {'(good ≥0.85)' if abs(rc) >= 0.85 else '(fair 0.65-0.85)' if abs(rc) >= 0.65 else '(poor <0.65)'}")

    # ── A4: Ceiling effect ────────────────────────────────────────────────
    human_at_max = [(human[c] == 7).mean() * 100 for c in HUMAN_SUB_COLS]
    llm_at_max = [(llm_rep[c] == 7).mean() * 100 for c in LLM_SUB_COLS]

    print("\n=== A4: Ceiling Effect (% at 7) ===")
    print(f"  {'Component':<12} {'Human%':>8} {'LLM%':>8} {'Delta':>8}")
    print(f"  {'-'*40}")
    for label, h_pct, l_pct in zip(SHORT_LABELS, human_at_max, llm_at_max):
        print(f"  {label:<12} {h_pct:>7.1f}% {l_pct:>7.1f}% {l_pct - h_pct:>+7.1f}%")

    # ── A4 continued: PCA with 9 components (drop fidelity pair) ─────────
    HUMAN_SUB_9 = [c for i, c in enumerate(HUMAN_SUB_COLS) if i not in FIDELITY_IDX]
    LLM_SUB_9 = [c for i, c in enumerate(LLM_SUB_COLS) if i not in FIDELITY_IDX]

    pca_h9, _ = run_pca(human[HUMAN_SUB_9].values.astype(float))
    pca_l9, _ = run_pca(llm_rep[LLM_SUB_9].values.astype(float))

    loadings_h9 = pca_h9.components_.T
    loadings_l9 = pca_l9.components_.T
    loadings_l9_aligned = sign_align(loadings_h9, loadings_l9)
    congruence_9 = tucker_congruence(loadings_h9, loadings_l9_aligned)

    print("\n=== A4: Tucker's Congruence (9 components, no fidelity) ===")
    for i, rc in enumerate(congruence_9[:4]):
        print(f"  PC{i+1}: rc={rc:.3f}  {'(good ≥0.85)' if abs(rc) >= 0.85 else '(fair 0.65-0.85)' if abs(rc) >= 0.65 else '(poor <0.65)'}")

    print(f"\n  Variance explained PC1+PC2: 11-comp={sum(pca_h.explained_variance_ratio_[:2]):.1%} (H) / {sum(pca_l.explained_variance_ratio_[:2]):.1%} (L)")
    print(f"  Variance explained PC1+PC2:  9-comp={sum(pca_h9.explained_variance_ratio_[:2]):.1%} (H) / {sum(pca_l9.explained_variance_ratio_[:2]):.1%} (L)")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_scree(pca_h, pca_l, suffix="_11comp")
    plot_loadings(loadings_h[:, :2], loadings_l_aligned[:, :2], SHORT_LABELS, congruence, suffix="_11comp")
    plot_scree(pca_h9, pca_l9, suffix="_9comp")
    plot_loadings(loadings_h9[:, :2], loadings_l9_aligned[:, :2], SHORT_LABELS_9, congruence_9, suffix="_9comp")
    plot_ceiling(human_at_max, llm_at_max, SHORT_LABELS)

    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
