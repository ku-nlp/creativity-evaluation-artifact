"""
Analyze prompt sensitivity results.
Reads: study2_llm/prompt_sensitivity/results/*.json
Outputs: figures + summary table
"""

import sys, os, json, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

OUT_DIR = "experiments_v2/prompt_sensitivity"

# ── Load results ──────────────────────────────────────────────────────────────
files = glob.glob("study2_llm/prompt_sensitivity/results/*.json")
if not files:
    print("No results found. Run run_sensitivity.py first.")
    sys.exit(1)

rows = []
for f in files:
    rows.extend(json.load(open(f)))

df = pd.DataFrame(rows)
df["IC_RC_diff"] = df["RC"] - df["IC"]
df["flipped"] = df["IC"] != df["RC"]

VARIANT_LABELS = {
    "v1": "V1: Neutral\n(both valid)",
    "v2": "V2: Minimal\n(no mention of change)",
    "v3": "V3: Final\n(natural to shift)",
    "v4": "V4: Original\n(has your view changed?)",
}

print("=" * 60)
print("PROMPT SENSITIVITY ANALYSIS")
print("=" * 60)

# ── Summary table ─────────────────────────────────────────────────────────────
summary = df.groupby("variant").agg(
    n          = ("story_id", "count"),
    IC_mean    = ("IC", "mean"),
    RC_mean    = ("RC", "mean"),
    flip_rate  = ("flipped", "mean"),
    mean_shift = ("IC_RC_diff", "mean"),
    std_shift  = ("IC_RC_diff", "std"),
).round(3)
summary["flip_rate_pct"] = (summary["flip_rate"] * 100).round(1)
print("\nSummary by variant:")
print(summary[["n", "IC_mean", "RC_mean", "flip_rate_pct", "mean_shift", "std_shift"]].to_string())

# ── RC variance across variants per story (key sensitivity measure) ────────────
pivot_rc = df.pivot_table(index="story_id", columns="variant", values="RC")
pivot_rc["rc_range"] = pivot_rc.max(axis=1) - pivot_rc.min(axis=1)
pivot_rc["rc_std"]   = pivot_rc[["v1","v2","v3","v4"]].std(axis=1)

print("\nRC variance across variants per story (std):")
print(pivot_rc[["v1","v2","v3","v4","rc_range","rc_std"]].sort_values("rc_std", ascending=False).to_string())
print(f"\nMean RC std across stories: {pivot_rc['rc_std'].mean():.3f}")
print(f"Max RC range across stories: {pivot_rc['rc_range'].max():.1f}")

# ── Pairwise correlation of RC scores between variants ────────────────────────
print("\nPairwise Pearson r of RC scores between variants:")
rc_corr = pivot_rc[["v1","v2","v3","v4"]].corr()
print(rc_corr.round(3).to_string())

# ── Figures ───────────────────────────────────────────────────────────────────

# Fig 1: Flip rate per variant
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

flip_rates = df.groupby("variant")["flipped"].mean() * 100
ax = axes[0]
colors = ["#1f77b4" if v != "v3" else "#d62728" for v in flip_rates.index]
ax.bar([VARIANT_LABELS[v] for v in flip_rates.index], flip_rates.values, color=colors)
ax.axhline(37, color="green", linestyle="--", lw=1.2, label="Human avg (27–47%)")
ax.axhline(27, color="green", linestyle=":", lw=0.8)
ax.axhline(47, color="green", linestyle=":", lw=0.8)
ax.set_ylabel("Flip Rate (%)")
ax.set_title("Flip Rate by Variant")
ax.set_ylim(0, 100)
ax.legend(fontsize=8)

# Fig 2: IC→RC shift distribution per variant
ax = axes[1]
variant_order = ["v1", "v2", "v3", "v4"]
shift_data = [df[df["variant"] == v]["IC_RC_diff"].values for v in variant_order]
ax.boxplot(shift_data, labels=[VARIANT_LABELS[v] for v in variant_order], patch_artist=True,
           boxprops=dict(facecolor="lightblue"))
ax.axhline(0, color="red", linestyle="--", lw=0.8)
ax.set_ylabel("RC − IC")
ax.set_title("IC→RC Shift Distribution")
ax.tick_params(axis="x", labelsize=7)

# Fig 3: RC per story per variant (heatmap)
ax = axes[2]
pivot_plot = pivot_rc[["v1","v2","v3","v4"]]
sns.heatmap(pivot_plot, ax=ax, cmap="RdYlGn", center=4, vmin=1, vmax=7,
            annot=True, fmt=".0f", linewidths=0.3,
            xticklabels=[VARIANT_LABELS[v] for v in ["v1","v2","v3","v4"]])
ax.set_title("RC per Story × Variant")
ax.tick_params(axis="x", labelsize=7)
ax.tick_params(axis="y", labelsize=7)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/sensitivity_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUT_DIR}/sensitivity_summary.png")

# Fig 2: Per-story RC variation across variants
fig, ax = plt.subplots(figsize=(10, 4))
story_order = pivot_rc["rc_std"].sort_values(ascending=False).index
x = range(len(story_order))
for v in ["v1","v2","v3","v4"]:
    ax.plot(x, pivot_rc.loc[story_order, v], marker="o", label=VARIANT_LABELS[v], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(story_order, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("RC Score")
ax.set_title("RC per Story by Variant (sorted by variability)")
ax.legend(fontsize=8, loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/sensitivity_per_story.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR}/sensitivity_per_story.png")

print("\n✓ Sensitivity analysis complete.")
print("\nInterpretation guide:")
print("  Mean RC std < 0.5  → low sensitivity, prompt is robust")
print("  Mean RC std > 1.0  → high sensitivity, wording matters")
print("  Flip rates close   → model behaviour stable across variants")
print("  Flip rates spread  → model responds to framing")
