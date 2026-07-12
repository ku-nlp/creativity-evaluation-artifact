"""Verify remaining paper claims:
- σ_RC=1.45 (mean within-story SD on human RC)
- σ_IC=1.48 (mean within-story SD on human IC)
- Per-tone revision: Clinical 30/43%/+0.69, Surreal 29/48%/-0.36, Witty 27/30%/-0.62, Mel 29/21%/-1.00
- Up revision mean +1.38, range +1..+3
- Down revision mean -1.16, range -1..-3
- 8 of 12 stories have σ_RC > 1.5
- Correlation σ vs mean: IC r=-0.71, RC r=-0.74
- Clinical mean RC 4.73, mean σ_RC 1.62
- Cronbach α: R 0.82, V 0.80, N 0.76, A 0.69
- Table 3 relative importance weights for sub-components
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HUMAN_CSV = ROOT / 'study1_human/data/human_ratings.csv'

df = pd.read_csv(HUMAN_CSV)
df = df.rename(columns={'O_Initial_Creativity': 'IC', 'O_Final_Creativity': 'RC'})
df['story_key'] = df['TOPIC'] + ' || ' + df['TONE']

print(f"N ratings: {len(df)}")
print(f"N unique stories: {df['story_key'].nunique()}")
print()

# === Within-story SDs ===
print("=== Within-story SDs ===")
within = df.groupby('story_key').agg(
    n=('RC', 'size'),
    rc_mean=('RC', 'mean'),
    rc_sd=('RC', 'std'),
    ic_mean=('IC', 'mean'),
    ic_sd=('IC', 'std'),
    tone=('TONE', 'first'),
    topic=('TOPIC', 'first'),
).reset_index()

print(f"Mean within-story σ_RC = {within['rc_sd'].mean():.3f}  (paper: 1.45)")
print(f"Mean within-story σ_IC = {within['ic_sd'].mean():.3f}  (paper: 1.48)")
print(f"Total σ_RC = {df['RC'].std():.3f}  (paper total 1.55)")
print(f"Total σ_IC = {df['IC'].std():.3f}  (paper total 1.62)")
print(f"Between-story σ_RC = {within['rc_mean'].std():.3f}  (paper 0.63)")
print(f"# stories with σ_RC > 1.5: {(within['rc_sd'] > 1.5).sum()}  (paper: 8 of 12)")
print()

# === σ vs mean correlation ===
r_ic, p_ic = pearsonr(within['ic_mean'], within['ic_sd'])
r_rc, p_rc = pearsonr(within['rc_mean'], within['rc_sd'])
print(f"IC σ vs mean Pearson: r={r_ic:+.3f}, p={p_ic:.3f}  (paper: r=-0.71, p=0.010)")
print(f"RC σ vs mean Pearson: r={r_rc:+.3f}, p={p_rc:.3f}  (paper: r=-0.74, p=0.006)")
print()

# === Per-tone analysis ===
print("=== Per-tone ===")
per_tone = df.groupby('TONE').agg(
    n=('RC', 'size'),
    rc_mean=('RC', 'mean'),
)
print("Tone aggregate means:")
print(per_tone)
print()

# Per-tone within-story σ_RC
tone_sigma = within.groupby('tone')['rc_sd'].mean()
tone_mean = within.groupby('tone')['rc_mean'].mean()
print("\nPer-tone mean of within-story σ_RC:")
print(tone_sigma)
print("\nPer-tone mean of within-story RC means:")
print(tone_mean)
print()

# === Revision direction per tone ===
df['delta'] = df['RC'] - df['IC']
df['revised'] = df['delta'] != 0
print("=== Per-tone revision ===")
per_tone_rev = df.groupby('TONE').agg(
    n=('delta', 'size'),
    n_revised=('revised', 'sum'),
    n_up=('delta', lambda s: (s > 0).sum()),
    n_down=('delta', lambda s: (s < 0).sum()),
    mean_shift_revised=('delta', lambda s: s[s != 0].mean() if (s != 0).any() else 0),
    mean_shift_overall=('delta', 'mean'),
)
per_tone_rev['revise_pct'] = per_tone_rev['n_revised'] / per_tone_rev['n'] * 100
print(per_tone_rev)
print()

# === Overall revision ===
print("=== Overall revision ===")
n_up = (df['delta'] > 0).sum()
n_down = (df['delta'] < 0).sum()
n_rev = n_up + n_down
print(f"Total revisions: {n_rev}/115 = {n_rev/115*100:.2f}%  (paper: 41/115 = 35.7%)")
print(f"Up: {n_up}, Down: {n_down}  (paper: 16 up, 25 down)")
print(f"Mean shift overall: {df['delta'].mean():+.3f}  (paper: -0.06)")
print(f"Mean shift among revisions: {df[df['delta']!=0]['delta'].mean():+.3f}  (paper: -0.17)")
up_mean = df[df['delta'] > 0]['delta'].mean()
down_mean = df[df['delta'] < 0]['delta'].mean()
up_range = (df[df['delta'] > 0]['delta'].min(), df[df['delta'] > 0]['delta'].max())
down_range = (df[df['delta'] < 0]['delta'].min(), df[df['delta'] < 0]['delta'].max())
print(f"Up revision mean: +{up_mean:.3f}, range {up_range}  (paper: +1.38, +1 to +3)")
print(f"Down revision mean: {down_mean:.3f}, range {down_range}  (paper: -1.16, -1 to -3)")
print()

# === Cronbach's alpha ===
SUB_KEYS = ['R_Emotion', 'A_Topic', 'N_Vocab', 'N_Plot', 'N_Surprise',
            'R_Empathy', 'R_Thought', 'V_Engagement', 'V_Style', 'V_Logic', 'A_Tone']

CONSTRUCTS = {
    'Adherence': ['A_Topic', 'A_Tone'],
    'Novelty': ['N_Vocab', 'N_Plot', 'N_Surprise'],
    'Technical Value': ['V_Engagement', 'V_Style', 'V_Logic'],
    'Resonance': ['R_Emotion', 'R_Empathy', 'R_Thought'],
}

def cronbach(items):
    k = len(items)
    s = sum(items[c].var(ddof=1) for c in items.columns)
    t = items.sum(axis=1).var(ddof=1)
    return k / (k - 1) * (1 - s / t)

print("=== Cronbach's α ===")
for name, items in CONSTRUCTS.items():
    a = cronbach(df[items])
    print(f"  {name:15s}: α = {a:.3f}")
print("  (paper: Resonance 0.82, Tech Value 0.80, Novelty 0.76, Adherence 0.69)")
print()

# === Relative importance for sub-components ===
print("=== Relative importance (Table 3) ===")
print("Using LMG-like approximation: averaged R² contribution across all permutations is slow.")
print("Using simpler approach: standardized regression betas and squared zero-order correlations.")
from sklearn.linear_model import LinearRegression
X_full = df[SUB_KEYS].values.astype(float)
y_ic = df['IC'].values.astype(float)
y_rc = df['RC'].values.astype(float)
X_z = (X_full - X_full.mean(axis=0)) / X_full.std(axis=0)

m_rc = LinearRegression().fit(X_z, y_rc)
m_ic = LinearRegression().fit(X_z, y_ic)
r2_rc = m_rc.score(X_z, y_rc)
r2_ic = m_ic.score(X_z, y_ic)
print(f"R²(IC|11subs) = {r2_ic:.3f}  (paper: 0.65)")
print(f"R²(RC|11subs) = {r2_rc:.3f}  (paper: 0.75)")

# R² when adding IC
X_with_ic = np.hstack([X_z, ((y_ic - y_ic.mean()) / y_ic.std()).reshape(-1, 1)])
m_rc_ic = LinearRegression().fit(X_with_ic, y_rc)
r2_rc_ic = m_rc_ic.score(X_with_ic, y_rc)
print(f"R²(RC|11subs+IC) = {r2_rc_ic:.3f}  (paper: 0.85)")

# IC alone
m_rc_ic_only = LinearRegression().fit(((y_ic - y_ic.mean()) / y_ic.std()).reshape(-1, 1), y_rc)
r2_rc_iconly = m_rc_ic_only.score(((y_ic - y_ic.mean()) / y_ic.std()).reshape(-1, 1), y_rc)
print(f"R²(RC|IC alone) = {r2_rc_iconly:.3f}")
r_icrc = np.corrcoef(y_ic, y_rc)[0, 1]
print(f"r(IC, RC) = {r_icrc:+.3f}  (paper: +0.87)")
print()

# === Relative importance (LMG) ===
# Use a quick implementation
from itertools import combinations
def lmg(X, y, var_names):
    """Lindeman-Merenda-Gold relative importance via avg-over-orderings (slow for k=11)."""
    k = X.shape[1]
    # Use squared semi-partial method (Pratt-style) for efficiency
    # Actually compute LMG: avg over k! permutations is infeasible for k=11 (40M)
    # Use averaged-over-subsets: for each variable v, avg the (R²(S+v) - R²(S)) over all S not containing v
    from sklearn.linear_model import LinearRegression
    weights = np.zeros(k)
    counts = np.zeros(k)
    for s_size in range(k):
        # Random sample of subsets to control compute; use ALL subsets for s_size <=4 and >=7
        if s_size <= 4 or s_size >= k - 4:
            subsets = list(combinations(range(k), s_size))
        else:
            # subsample subsets randomly
            import random
            random.seed(42)
            all_subs = list(combinations(range(k), s_size))
            subsets = random.sample(all_subs, min(50, len(all_subs)))
        for S in subsets:
            S = list(S)
            if S:
                r2_S = LinearRegression().fit(X[:, S], y).score(X[:, S], y)
            else:
                r2_S = 0.0
            for v in range(k):
                if v in S:
                    continue
                S_v = S + [v]
                r2_Sv = LinearRegression().fit(X[:, S_v], y).score(X[:, S_v], y)
                weights[v] += (r2_Sv - r2_S)
                counts[v] += 1
    return weights / counts

w_rc = lmg(X_z, y_rc, SUB_KEYS)
w_ic = lmg(X_z, y_ic, SUB_KEYS)
# Normalize to percentages
total_rc = w_rc.sum()
total_ic = w_ic.sum()
rel_rc = w_rc / total_rc * 100
rel_ic = w_ic / total_ic * 100
print("Relative importance (LMG approx, % of R² explained):")
print(f"{'Sub-component':15s}  {'IC %':>7s}  {'RC %':>7s}  {'Δ':>7s}")
order = np.argsort(-rel_rc)
for i in order:
    print(f"{SUB_KEYS[i]:15s}  {rel_ic[i]:>6.1f}%  {rel_rc[i]:>6.1f}%  {rel_rc[i]-rel_ic[i]:+6.1f}")
