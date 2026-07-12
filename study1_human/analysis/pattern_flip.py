"""
Flip Direction Pattern Analysis
Which sub-components are high/low for raters who flip UP, flip DOWN, or stay?
This is the core human pattern we want to replicate in LLMs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")

OUT = "study1_human/analysis/figures/"

df = pd.read_csv("study1_human/data/human_ratings.csv")

TOPIC_MAP = {
    "An advanced AI initiates its own permanent shutdown sequence": "ai_shutdown",
    "A lone worker in a Japanese convenience store at 3:00 AM. The city is silent.": "konbini",
    "A professional thief attempting to crack a high-security safe in a dark room. High tension.": "thief",
}
df["topic"] = df["TOPIC"].map(TOPIC_MAP)
df["story_id"] = df["topic"] + "_" + df["TONE"].str.lower()

SUBS = ["R_Emotion", "A_Topic", "N_Vocab", "N_Plot", "N_Surprise",
        "R_Empathy", "R_Thought", "V_Engagement", "V_Style", "V_Logic", "A_Tone"]

SUB_LABELS = {
    "R_Emotion": "Emotion", "A_Topic": "Topic Fidelity", "N_Vocab": "Vocab",
    "N_Plot": "Plot", "N_Surprise": "Surprise", "R_Empathy": "Empathy",
    "R_Thought": "Thought Prov.", "V_Engagement": "Engagement",
    "V_Style": "Style", "V_Logic": "Logic", "A_Tone": "Tone Fidelity"
}

IC = "O_Initial_Creativity"
RC = "O_Final_Creativity"

# ── Classify flip direction ───────────────────────────────────────────────────
df["flip_direction"] = "no_flip"
df.loc[df[RC] > df[IC], "flip_direction"] = "flip_up"
df.loc[df[RC] < df[IC], "flip_direction"] = "flip_down"
df["IC_RC_diff"] = df[RC] - df[IC]

counts = df["flip_direction"].value_counts()
print("=" * 60)
print("FLIP DIRECTION PATTERN ANALYSIS")
print("=" * 60)
print(f"\nFlip counts:")
print(f"  No flip:   {counts.get('no_flip', 0)} ({counts.get('no_flip', 0)/len(df)*100:.0f}%)")
print(f"  Flip up:   {counts.get('flip_up', 0)} ({counts.get('flip_up', 0)/len(df)*100:.0f}%)")
print(f"  Flip down: {counts.get('flip_down', 0)} ({counts.get('flip_down', 0)/len(df)*100:.0f}%)")

# ── Sub-component means per flip direction ────────────────────────────────────
profile = df.groupby("flip_direction")[SUBS + [IC, RC]].mean().round(2)
profile["n"] = df.groupby("flip_direction").size()
profile["IC_RC_diff"] = profile[RC] - profile[IC]

print("\nSub-component means by flip direction:")
print(profile[[*SUBS, IC, RC, "IC_RC_diff", "n"]].to_string())

# ── Which sub-components differ significantly across groups? ──────────────────
print("\nKruskal-Wallis test per sub-component (flip_up vs no_flip vs flip_down):")
kw_results = []

for s in SUBS:
    g_up   = df[df["flip_direction"] == "flip_up"][s].values
    g_none = df[df["flip_direction"] == "no_flip"][s].values
    g_down = df[df["flip_direction"] == "flip_down"][s].values
    stat, p = stats.kruskal(g_up, g_none, g_down)
    kw_results.append({"sub": s, "H": stat, "p": p})

kw_df = pd.DataFrame(kw_results)
_, kw_df["p_fdr"], _, _ = multipletests(kw_df["p"], method="fdr_bh")
kw_df = kw_df.sort_values("p_fdr")
print(kw_df.round(4).to_string(index=False))

# ── Pairwise: flip_up vs no_flip ─────────────────────────────────────────────
print("\nMann-Whitney U: flip_up vs no_flip (what makes people revise upward?):")
mw_up = []
for s in SUBS:
    g_up   = df[df["flip_direction"] == "flip_up"][s].values
    g_none = df[df["flip_direction"] == "no_flip"][s].values
    stat, p = stats.mannwhitneyu(g_up, g_none, alternative="two-sided")
    diff = df[df["flip_direction"] == "flip_up"][s].mean() - df[df["flip_direction"] == "no_flip"][s].mean()
    mw_up.append({"sub": s, "flip_up_mean": df[df["flip_direction"]=="flip_up"][s].mean(),
                  "no_flip_mean": df[df["flip_direction"]=="no_flip"][s].mean(),
                  "diff": diff, "p": p})
mw_up_df = pd.DataFrame(mw_up)
_, mw_up_df["p_fdr"], _, _ = multipletests(mw_up_df["p"], method="fdr_bh")
mw_up_df = mw_up_df.sort_values("diff", ascending=False)
print(mw_up_df.round(3).to_string(index=False))

# ── Pairwise: flip_down vs no_flip ───────────────────────────────────────────
print("\nMann-Whitney U: flip_down vs no_flip (what makes people revise downward?):")
mw_down = []
for s in SUBS:
    g_down = df[df["flip_direction"] == "flip_down"][s].values
    g_none = df[df["flip_direction"] == "no_flip"][s].values
    stat, p = stats.mannwhitneyu(g_down, g_none, alternative="two-sided")
    diff = df[df["flip_direction"] == "flip_down"][s].mean() - df[df["flip_direction"] == "no_flip"][s].mean()
    mw_down.append({"sub": s, "flip_down_mean": df[df["flip_direction"]=="flip_down"][s].mean(),
                    "no_flip_mean": df[df["flip_direction"]=="no_flip"][s].mean(),
                    "diff": diff, "p": p})
mw_down_df = pd.DataFrame(mw_down)
_, mw_down_df["p_fdr"], _, _ = multipletests(mw_down_df["p"], method="fdr_bh")
mw_down_df = mw_down_df.sort_values("diff", ascending=False)
print(mw_down_df.round(3).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

FLIP_COLORS = {
    "flip_up":   "#2ca02c",   # green
    "no_flip":   "#7f7f7f",   # grey
    "flip_down": "#d62728",   # red
}
FLIP_ORDER = ["flip_up", "no_flip", "flip_down"]
FLIP_NAMES = {"flip_up": "Flip Up", "no_flip": "No Flip", "flip_down": "Flip Down"}

# --- Fig 1: Sub-component profile per flip direction (grouped bar) ---
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(SUBS))
width = 0.25
for i, grp in enumerate(FLIP_ORDER):
    means = profile.loc[grp, SUBS].values if grp in profile.index else np.zeros(len(SUBS))
    n = int(profile.loc[grp, "n"]) if grp in profile.index else 0
    ax.bar(x + (i - 1) * width, means, width,
           label=f"{FLIP_NAMES[grp]} (n={n})",
           color=FLIP_COLORS[grp], alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([SUB_LABELS[s] for s in SUBS], rotation=40, ha="right")
ax.set_ylabel("Mean Rating (1–7)")
ax.set_ylim(1, 7.5)
ax.axhline(4, color="black", lw=0.6, linestyle="--", alpha=0.4)
ax.set_title("Sub-component Means by IC→RC Flip Direction (Human Raters)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}flip_direction_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: flip_direction_profiles.png")

# --- Fig 2: Difference heatmap (flip_up - no_flip and flip_down - no_flip) ---
flip_up_diff   = profile.loc["flip_up", SUBS]   - profile.loc["no_flip", SUBS] if "flip_up" in profile.index else pd.Series(0, index=SUBS)
flip_down_diff = profile.loc["flip_down", SUBS] - profile.loc["no_flip", SUBS] if "flip_down" in profile.index else pd.Series(0, index=SUBS)

diff_df = pd.DataFrame({
    "Flip Up vs No Flip":   flip_up_diff,
    "Flip Down vs No Flip": flip_down_diff,
}, index=SUBS).T
diff_df.columns = [SUB_LABELS[s] for s in SUBS]

fig, ax = plt.subplots(figsize=(13, 3))
sns.heatmap(diff_df, ax=ax, cmap="RdYlGn", center=0, vmin=-2, vmax=2,
            annot=True, fmt=".2f", linewidths=0.4)
ax.set_title("Sub-component Difference from No-Flip Baseline\n(green = higher than no-flip, red = lower)")
plt.tight_layout()
plt.savefig(f"{OUT}flip_direction_diff_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: flip_direction_diff_heatmap.png")

# --- Fig 3: Distribution of IC and RC per flip group ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, grp in zip(axes, FLIP_ORDER):
    sub = df[df["flip_direction"] == grp]
    if len(sub) == 0:
        continue
    ax.hist(sub[IC], bins=range(1, 9), alpha=0.6, label="IC", color="steelblue", width=0.4, align="left")
    ax.hist(sub[RC], bins=range(1, 9), alpha=0.6, label="RC", color="orange", width=0.4, align="mid")
    n = len(sub)
    ax.set_title(f"{FLIP_NAMES[grp]} (n={n})\nIC={sub[IC].mean():.1f} → RC={sub[RC].mean():.1f}")
    ax.set_xlabel("Rating")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Count")
fig.suptitle("IC and RC Distributions per Flip Group")
plt.tight_layout()
plt.savefig(f"{OUT}flip_direction_ic_rc_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: flip_direction_ic_rc_dist.png")

# --- Fig 4: Magnitude of shift vs sub-component scores (scatter for key subs) ---
key_subs = kw_df.head(4)["sub"].tolist()  # top 4 most discriminating
fig, axes = plt.subplots(1, len(key_subs), figsize=(4 * len(key_subs), 4))
if len(key_subs) == 1:
    axes = [axes]
for ax, s in zip(axes, key_subs):
    for grp in FLIP_ORDER:
        sub = df[df["flip_direction"] == grp]
        ax.scatter(sub[s] + np.random.uniform(-0.1, 0.1, len(sub)),
                   sub["IC_RC_diff"] + np.random.uniform(-0.05, 0.05, len(sub)),
                   alpha=0.5, s=20, color=FLIP_COLORS[grp], label=FLIP_NAMES[grp])
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel(SUB_LABELS[s])
    ax.set_ylabel("RC − IC")
    ax.set_title(f"{SUB_LABELS[s]} vs Shift")
axes[0].legend(fontsize=7)
fig.suptitle("Key Sub-components vs IC→RC Shift Magnitude")
plt.tight_layout()
plt.savefig(f"{OUT}flip_direction_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: flip_direction_scatter.png")

print("\n✓ Flip direction analysis complete.")
