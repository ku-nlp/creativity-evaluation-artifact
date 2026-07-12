"""
Phase 2A: Revisability by model family
Phase 2B: Reasoning mode paired comparisons

Produces:
  1. Family revisability summary (Wilcoxon IC vs RC per model, effect sizes)
  2. Logistic regression: flipped ~ family + tone + IC
  3. Reasoning mode paired comparisons (Wilcoxon per pair)
  4. Figures: family flip bar chart, reasoning paired dot plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

base = Path(__file__).resolve().parent.parent
llm = pd.read_csv(base / "data/llm_ratings.csv")

SUB_COMPONENTS = [
    "emotional_impact", "topic_fidelity", "vocabulary_freshness",
    "plot_uniqueness", "surprise", "empathy", "thought_provocation",
    "engagement", "stylistic_quality", "logical_coherence", "tone_fidelity",
]

STORY_ORDER = sorted(llm["story_id"].unique())

# =====================================================================
# PHASE 2A: Revisability by model family
# =====================================================================
print("=" * 70)
print("PHASE 2A: REVISABILITY BY MODEL FAMILY")
print("=" * 70)

# --- Per-model Wilcoxon signed-rank: IC vs RC (paired by story) ---
print("\n--- Wilcoxon Signed-Rank: IC vs RC per Model (paired by story) ---")
model_results = []
for model in llm["model_label"].unique():
    mdf = llm[llm["model_label"] == model].sort_values("story_id")
    ic = mdf["IC"].values
    rc = mdf["RC"].values
    diff = rc - ic
    n_nonzero = np.sum(diff != 0)

    if n_nonzero >= 2:
        w_stat, p_val = stats.wilcoxon(diff[diff != 0])
        # Rank-biserial correlation (effect size for Wilcoxon)
        r_pos = np.sum(np.where(diff > 0, np.abs(diff), 0))
        r_neg = np.sum(np.where(diff < 0, np.abs(diff), 0))
        rbc = (r_pos - r_neg) / (r_pos + r_neg) if (r_pos + r_neg) > 0 else 0
    else:
        w_stat, p_val = np.nan, np.nan
        rbc = 0.0

    family = mdf["family"].iloc[0]
    model_results.append({
        "model": model, "family": family,
        "mean_IC": ic.mean(), "mean_RC": rc.mean(),
        "mean_flip": diff.mean(), "n_flips": np.sum(diff != 0),
        "n_up": np.sum(diff > 0), "n_down": np.sum(diff < 0),
        "W": w_stat, "p": p_val, "rbc": rbc,
    })

model_df = pd.DataFrame(model_results).sort_values("n_flips", ascending=False)

print(f"  {'Model':35s} {'Family':10s} {'Flips':>6s} {'Mean':>7s} {'W':>7s} {'p':>8s} {'rbc':>6s}")
for _, r in model_df.iterrows():
    sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
    p_str = f"{r['p']:.4f}" if not np.isnan(r["p"]) else "   N/A"
    w_str = f"{r['W']:.0f}" if not np.isnan(r["W"]) else "  N/A"
    print(f"  {r['model']:35s} {r['family']:10s} {r['n_flips']:>3.0f}/12 "
          f"{r['mean_flip']:+6.2f} {w_str:>7s} {p_str:>8s} {r['rbc']:+5.2f} {sig}")

# --- Per-family aggregated ---
print("\n--- Family-Level Summary ---")
family_summary = llm.groupby("family").agg(
    n_models=("model_label", "nunique"),
    n_rows=("story_id", "count"),
    mean_IC=("IC", "mean"),
    mean_RC=("RC", "mean"),
    flip_rate=("flipped", "mean"),
    mean_flip=("gk_flip", "mean"),
    n_flips=("flipped", "sum"),
).sort_values("flip_rate", ascending=False)

for fam, r in family_summary.iterrows():
    print(f"  {fam:12s}  models={r['n_models']:.0f}  N={r['n_rows']:.0f}  "
          f"flip_rate={r['flip_rate']:.0%}  mean_flip={r['mean_flip']:+.3f}  "
          f"IC={r['mean_IC']:.2f}  RC={r['mean_RC']:.2f}")

# --- Logistic regression: flipped ~ family + tone + IC ---
print("\n--- Logistic Regression: flipped ~ family + tone + IC ---")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

log_df = llm[["flipped", "family", "tone", "IC"]].copy()
# Encode categoricals
fam_dummies = pd.get_dummies(log_df["family"], prefix="fam", drop_first=True)
tone_dummies = pd.get_dummies(log_df["tone"], prefix="tone", drop_first=True)
X = pd.concat([fam_dummies, tone_dummies, log_df[["IC"]]], axis=1)
y = log_df["flipped"]

# Fit
logreg = LogisticRegression(max_iter=1000, penalty=None)
logreg.fit(X, y)

print(f"  Accuracy: {logreg.score(X, y):.1%}")
print(f"  {'Feature':25s} {'Coeff':>8s} {'Odds Ratio':>12s}")
for feat, coef in zip(X.columns, logreg.coef_[0]):
    print(f"  {feat:25s} {coef:+8.3f} {np.exp(coef):12.3f}")
print(f"  {'intercept':25s} {logreg.intercept_[0]:+8.3f}")

# McFadden's pseudo-R²
from sklearn.metrics import log_loss
ll_full = -log_loss(y, logreg.predict_proba(X), normalize=False)
ll_null = -log_loss(y, np.full((len(y), 2), [1 - y.mean(), y.mean()]), normalize=False)
pseudo_r2 = 1 - ll_full / ll_null
print(f"  McFadden's pseudo-R²: {pseudo_r2:.3f}")

# =====================================================================
# Figure 2A: Family flip rates
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 2A-1: Flip rate by family
ax = axes[0]
fam_order = family_summary.sort_values("flip_rate", ascending=True).index
colors = {"Meta": "#e74c3c", "Google": "#3498db", "Alibaba": "#2ecc71",
          "OpenAI": "#9b59b6", "Microsoft": "#f39c12"}
x = np.arange(len(fam_order))
bars = ax.barh(x, [family_summary.loc[f, "flip_rate"] * 100 for f in fam_order],
               color=[colors[f] for f in fam_order], edgecolor="white", alpha=0.85)
ax.set_yticks(x)
ax.set_yticklabels(fam_order, fontsize=11)
ax.set_xlabel("Flip Rate (%)", fontsize=11)
ax.set_title("Gatekeeper Flip Rate by Family", fontsize=13, fontweight="bold")
for i, f in enumerate(fam_order):
    rate = family_summary.loc[f, "flip_rate"]
    n = family_summary.loc[f, "n_flips"]
    total = family_summary.loc[f, "n_rows"]
    ax.text(rate * 100 + 1, i, f"{rate:.0%} ({n:.0f}/{total:.0f})", va="center", fontsize=9)
ax.set_xlim(0, 70)
ax.grid(True, axis="x", alpha=0.2)

# 2A-2: Per-model flip direction
ax = axes[1]
model_order = model_df.sort_values(["family", "n_flips"], ascending=[True, False])["model"].values
y_pos = np.arange(len(model_order))

for i, model in enumerate(model_order):
    r = model_df[model_df["model"] == model].iloc[0]
    ax.barh(i, r["n_up"], color="#2ecc71", edgecolor="white", alpha=0.8, height=0.7)
    ax.barh(i, -r["n_down"], color="#e74c3c", edgecolor="white", alpha=0.8, height=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(model_order, fontsize=7)
ax.set_xlabel("← Down flips | Up flips →", fontsize=10)
ax.set_title("Flip Direction by Model", fontsize=13, fontweight="bold")
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlim(-6, 10)
ax.grid(True, axis="x", alpha=0.2)

fig.tight_layout()
fig.savefig(OUT / "2a_family_revisability.png", dpi=200, bbox_inches="tight")
print(f"\nSaved: {OUT / '2a_family_revisability.png'}")

# =====================================================================
# PHASE 2B: Reasoning mode paired comparisons
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 2B: REASONING MODE PAIRED COMPARISONS")
print("=" * 70)

# Define paired comparisons
PAIRS = [
    ("Qwen 3 8B (thinking_off)", "Qwen 3 8B (thinking_on)", "Qwen 8B: thinking off vs on"),
    ("Qwen 3 32B (thinking_off)", "Qwen 3 32B (thinking_on)", "Qwen 32B: thinking off vs on"),
    ("GPT-4.1", "o4-mini (reasoning)", "OpenAI: GPT-4.1 vs o4-mini"),
    ("Gemini 2.5 Pro (budget=128)", "Gemini 2.5 Pro", "Gemini 2.5: budget=128 vs default"),
    ("Gemini 2.5 Pro", "Gemini 2.5 Pro (budget=32768)", "Gemini 2.5: default vs budget=32768"),
    ("Gemini 3.1 Pro (low)", "Gemini 3.1 Pro (medium)", "Gemini 3.1: low vs medium"),
    ("Gemini 3.1 Pro (medium)", "Gemini 3.1 Pro (high)", "Gemini 3.1: medium vs high"),
    ("Gemini 3.1 Pro (low)", "Gemini 3.1 Pro (high)", "Gemini 3.1: low vs high"),
]

COMPARE_COLS = ["IC", "RC", "gk_flip", "enjoyment", "sub_mean"]

pair_results = []
print(f"\n{'Comparison':45s} {'Measure':>10s} {'A_mean':>7s} {'B_mean':>7s} {'Δ':>6s} {'W':>6s} {'p':>8s}")
for model_a, model_b, label in PAIRS:
    a = llm[llm["model_label"] == model_a].sort_values("story_id")
    b = llm[llm["model_label"] == model_b].sort_values("story_id")
    if len(a) == 0 or len(b) == 0:
        print(f"  WARNING: {label} — missing data")
        continue

    for col in COMPARE_COLS:
        va = a[col].values
        vb = b[col].values
        diff = vb - va
        n_nonzero = np.sum(diff != 0)

        if n_nonzero >= 2:
            w, p = stats.wilcoxon(diff[diff != 0])
        else:
            w, p = np.nan, np.nan

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "" if not np.isnan(p) else ""
        p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
        w_str = f"{w:.0f}" if not np.isnan(w) else "N/A"
        print(f"  {label:43s} {col:>10s} {va.mean():7.2f} {vb.mean():7.2f} "
              f"{diff.mean():+5.2f} {w_str:>6s} {p_str:>8s} {sig}")

        pair_results.append({
            "comparison": label, "measure": col,
            "a_mean": va.mean(), "b_mean": vb.mean(),
            "delta": diff.mean(), "W": w, "p": p,
        })
    print()

# =====================================================================
# Figure 2B: Reasoning paired comparisons (Fig 4 candidate)
# =====================================================================
# Focus on IC and RC for the key pairs
key_pairs = [
    ("Qwen 3 8B (thinking_off)", "Qwen 3 8B (thinking_on)", "Qwen 8B"),
    ("Qwen 3 32B (thinking_off)", "Qwen 3 32B (thinking_on)", "Qwen 32B"),
    ("GPT-4.1", "o4-mini (reasoning)", "OpenAI"),
    ("Gemini 2.5 Pro (budget=128)", "Gemini 2.5 Pro (budget=32768)", "Gemini 2.5\n(min vs max)"),
    ("Gemini 3.1 Pro (low)", "Gemini 3.1 Pro (high)", "Gemini 3.1\n(low vs high)"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, measure, title in zip(axes, ["IC", "RC"], ["Initial Creativity", "Reflective Creativity"]):
    x = np.arange(len(key_pairs))
    width = 0.35

    a_vals = []
    b_vals = []
    for ma, mb, _ in key_pairs:
        a_vals.append(llm[llm["model_label"] == ma][measure].mean())
        b_vals.append(llm[llm["model_label"] == mb][measure].mean())

    ax.bar(x - width / 2, a_vals, width, label="Less reasoning", color="#3498db",
           edgecolor="white", alpha=0.85)
    ax.bar(x + width / 2, b_vals, width, label="More reasoning", color="#e74c3c",
           edgecolor="white", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([p[2] for p in key_pairs], fontsize=9)
    ax.set_ylabel(f"Mean {measure}", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_ylim(4.5, 7.5)

    # Add delta annotations
    for i in range(len(key_pairs)):
        delta = b_vals[i] - a_vals[i]
        y_pos = max(a_vals[i], b_vals[i]) + 0.1
        ax.text(i, y_pos, f"Δ={delta:+.2f}", ha="center", fontsize=8, fontweight="bold")

fig.suptitle("Reasoning Mode: Does More Thinking Change Scores?", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "2b_reasoning_comparison.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT / '2b_reasoning_comparison.png'}")

# --- Flip rate comparison ---
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(key_pairs))
width = 0.35

a_flips = []
b_flips = []
for ma, mb, _ in key_pairs:
    a_flips.append(llm[llm["model_label"] == ma]["flipped"].mean() * 100)
    b_flips.append(llm[llm["model_label"] == mb]["flipped"].mean() * 100)

ax.bar(x - width / 2, a_flips, width, label="Less reasoning", color="#3498db",
       edgecolor="white", alpha=0.85)
ax.bar(x + width / 2, b_flips, width, label="More reasoning", color="#e74c3c",
       edgecolor="white", alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([p[2] for p in key_pairs], fontsize=9)
ax.set_ylabel("Flip Rate (%)", fontsize=11)
ax.set_title("Reasoning Mode Does Not Enable Flipping", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.2)

for i in range(len(key_pairs)):
    delta = b_flips[i] - a_flips[i]
    y_pos = max(a_flips[i], b_flips[i]) + 2
    ax.text(i, y_pos, f"Δ={delta:+.0f}%", ha="center", fontsize=9, fontweight="bold")

fig.tight_layout()
fig.savefig(OUT / "2b_reasoning_flip_rate.png", dpi=200, bbox_inches="tight")
print(f"Saved: {OUT / '2b_reasoning_flip_rate.png'}")

print("\n" + "=" * 70)
print("Phase 2A+2B complete.")
print("=" * 70)
