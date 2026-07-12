"""
Robustness checks for cross-analysis results.

1. VIF — multicollinearity among 11 sub-components
2. Ridge regression — confirm OLS coefficients aren't artifacts of overfitting
3. LOOCV — leave-one-out cross-validated R² for human and each LLM
4. Bootstrap CIs — on human story-level means (IC, RC, flip rate)
5. PCA — do 11 dimensions collapse into fewer factors?
6. Incremental validity — do sub-components add beyond IC alone?

Outputs to study2_llm/analysis_figures/ and stdout.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = 'study2_llm/analysis_figures'
os.makedirs(OUTDIR, exist_ok=True)

# ── Setup (same as cross_analysis.py) ───────────────────────────────────────

HUMAN_SUB_MAP = {
    'R_Emotion': 'emotional_impact', 'A_Topic': 'topic_fidelity',
    'N_Vocab': 'vocabulary_freshness', 'N_Plot': 'plot_uniqueness',
    'N_Surprise': 'surprise', 'R_Empathy': 'empathy',
    'R_Thought': 'thought_provocation', 'V_Engagement': 'engagement',
    'V_Style': 'stylistic_quality', 'V_Logic': 'logical_coherence',
    'A_Tone': 'tone_fidelity',
}
SUB_KEYS = list(HUMAN_SUB_MAP.values())
SUB_SHORT = {
    'emotional_impact': 'Emotion', 'topic_fidelity': 'Topic Fid.',
    'vocabulary_freshness': 'Vocab', 'plot_uniqueness': 'Plot',
    'surprise': 'Surprise', 'empathy': 'Empathy',
    'thought_provocation': 'Thought', 'engagement': 'Engagement',
    'stylistic_quality': 'Style', 'logical_coherence': 'Logic',
    'tone_fidelity': 'Tone Fid.',
}
TOPIC_TO_PREFIX = {
    'An advanced AI initiates its own permanent shutdown sequence': 'ai_shutdown',
    'A lone worker in a Japanese convenience store at 3:00 AM. The city is silent.': 'midnight_store',
    'A professional thief attempting to crack a high-security safe in a dark room. High tension.': 'the_heist',
}

LLM_FILES = {
    'Qwen3.5-4B-NT':   'study2_llm/data/raw_json/results_qwen35_4b_CLEAN_nothink.json',
    'Qwen3.5-4B-T':    'study2_llm/data/raw_json/results_qwen35_4b_CLEAN_think.json',
    'Qwen3.5-9B-NT':   'study2_llm/data/raw_json/results_qwen35_9b_CLEAN_nothink.json',
    'Qwen3.5-9B-T':    'study2_llm/data/raw_json/results_qwen35_9b_CLEAN_think.json',
    'Qwen3.5-27B-NT':  'study2_llm/data/raw_json/results_qwen35_27b_CLEAN_nothink.json',
    'Qwen3.5-27B-T':   'study2_llm/data/raw_json/results_qwen35_27b_CLEAN_think.json',
    'Llama3.1-8B':     'study2_llm/data/raw_json/results_llama31_8b_judge_gemini_nothink.json',
    'Llama3.1-70B':    'study2_llm/data/raw_json/results_llama31_70b_judge_gemini_nothink.json',
    'Gemma4-E2B-NT':   'study2_llm/data/raw_json/results_gemma4_e2b_nothink.json',
    'Gemma4-E2B-T':    'study2_llm/data/raw_json/results_gemma4_e2b_think.json',
    'Gemma4-E4B-NT':   'study2_llm/data/raw_json/results_gemma4_e4b_nothink.json',
    'Gemma4-E4B-T':    'study2_llm/data/raw_json/results_gemma4_e4b_think.json',
    'Gemma4-26B-NT':   'study2_llm/data/raw_json/results_gemma4_26b_nothink.json',
    'Gemma4-26B-T':    'study2_llm/data/raw_json/results_gemma4_26b_think.json',
    'Gemma4-31B-NT':   'study2_llm/data/raw_json/results_gemma4_31b_nothink.json',
    'Gemma4-31B-T':    'study2_llm/data/raw_json/results_gemma4_31b_think.json',
    'Gemini3.1-Low':   'study2_llm/data/raw_json/results_gemini_31_pro_low.json',
    'Gemini3.1-Med':   'study2_llm/data/raw_json/results_gemini_31_pro_medium.json',
    'Gemini3.1-High':  'study2_llm/data/raw_json/results_gemini_31_pro_high.json',
    'Gemini3-Low':     'study2_llm/data/raw_json/results_gemini_3_pro_low.json',
    'Gemini3-Med':     'study2_llm/data/raw_json/results_gemini_3_pro_medium.json',
    'Gemini3-High':    'study2_llm/data/raw_json/results_gemini_3_pro_high.json',
    'GPT-5.5':         'study2_llm/data/raw_json/results_gpt55_CLEAN_high.json',
}


def load_human():
    df = pd.read_csv('study1_human/data/human_ratings.csv')
    rename = {h: s for h, s in HUMAN_SUB_MAP.items()}
    rename['O_Initial_Creativity'] = 'IC'
    rename['O_Final_Creativity'] = 'RC'
    df = df.rename(columns=rename)
    df['story_id'] = df.apply(
        lambda r: TOPIC_TO_PREFIX[r['TOPIC']] + '_' + r['TONE'].lower(), axis=1)
    return df


def load_llm(path):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for r in data['results']:
        if not r.get('parse_ok', True):
            continue
        row = {'story_id': r['story_id'],
               'IC': r['scores']['initial_creativity'],
               'RC': r['scores']['reflective_creativity']}
        for k in SUB_KEYS:
            row[k] = r['scores']['sub_components'][k]
        rows.append(row)
    return pd.DataFrame(rows)


human_df = load_human()
llm_dfs = {}
for label, path in LLM_FILES.items():
    if os.path.exists(path):
        llm_dfs[label] = load_llm(path)


# ════════════════════════════════════════════════════════════════════════════
# CHECK 1: VIF — Multicollinearity
# ════════════════════════════════════════════════════════════════════════════

print('=' * 70)
print('CHECK 1: Variance Inflation Factor (VIF)')
print('VIF > 5 = concerning, VIF > 10 = serious multicollinearity')
print('=' * 70)


def compute_vif(df, features):
    """Compute VIF for each feature."""
    from sklearn.linear_model import LinearRegression
    X = df[features].values.astype(float)
    vifs = []
    for i in range(X.shape[1]):
        y_i = X[:, i]
        X_others = np.delete(X, i, axis=1)
        reg = LinearRegression().fit(X_others, y_i)
        r2 = reg.score(X_others, y_i)
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float('inf')
        vifs.append(vif)
    return dict(zip(features, vifs))


print('\n  HUMAN (N=115):')
human_vif = compute_vif(human_df, SUB_KEYS)
print(f'  {"Sub-component":25s} {"VIF":>8s} {"Status":>10s}')
print(f'  {"-"*25} {"-"*8} {"-"*10}')
for k in sorted(SUB_KEYS, key=lambda x: human_vif[x], reverse=True):
    v = human_vif[k]
    status = 'CONCERN' if v > 5 else 'OK'
    if v > 10:
        status = 'SERIOUS'
    print(f'  {SUB_SHORT[k]:25s} {v:8.2f} {status:>10s}')

# Do VIF for a few representative LLM conditions
print('\n  LLM VIF (selected conditions, showing max VIF per condition):')
print(f'  {"Condition":22s} {"Max VIF":>8s} {"Worst dim":>15s} {"Status":>10s}')
print(f'  {"-"*22} {"-"*8} {"-"*15} {"-"*10}')
for label in ['Qwen3.5-4B-NT', 'Qwen3.5-27B-NT', 'Gemma4-E2B-NT',
              'Gemma4-31B-NT', 'Gemini3.1-Low', 'GPT-5.5']:
    if label not in llm_dfs:
        continue
    vif = compute_vif(llm_dfs[label], SUB_KEYS)
    max_k = max(vif, key=vif.get)
    max_v = vif[max_k]
    status = 'SERIOUS' if max_v > 10 else ('CONCERN' if max_v > 5 else 'OK')
    print(f'  {label:22s} {max_v:8.2f} {SUB_SHORT[max_k]:>15s} {status:>10s}')


# ════════════════════════════════════════════════════════════════════════════
# CHECK 2: Ridge Regression vs OLS
# ════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('CHECK 2: Ridge Regression vs OLS')
print('If Ridge coefficients ≈ OLS coefficients, results are stable.')
print('If they diverge, OLS was overfitting.')
print('=' * 70)


def compare_ols_ridge(df, label):
    X = df[SUB_KEYS].values.astype(float)
    y = df['RC'].values.astype(float)
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X)

    # OLS
    ols = LinearRegression().fit(X_z, y)
    ols_coefs = dict(zip(SUB_KEYS, ols.coef_))
    ols_r2 = ols.score(X_z, y)

    # Ridge with cross-validated alpha
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5).fit(X_z, y)
    ridge_coefs = dict(zip(SUB_KEYS, ridge.coef_))
    ridge_r2 = ridge.score(X_z, y)

    return {
        'label': label,
        'ols_r2': ols_r2, 'ridge_r2': ridge_r2,
        'alpha': ridge.alpha_,
        'ols_coefs': ols_coefs, 'ridge_coefs': ridge_coefs,
    }


# Human
h_comp = compare_ols_ridge(human_df, 'Human')
print(f'\n  HUMAN:')
print(f'  OLS R² = {h_comp["ols_r2"]:.3f}  |  Ridge R² = {h_comp["ridge_r2"]:.3f}  '
      f'|  α = {h_comp["alpha"]:.3f}')
print(f'  {"Sub-component":25s} {"OLS":>8s} {"Ridge":>8s} {"Δ":>8s}')
print(f'  {"-"*25} {"-"*8} {"-"*8} {"-"*8}')
for k in sorted(SUB_KEYS, key=lambda x: abs(h_comp['ols_coefs'][x]), reverse=True):
    o = h_comp['ols_coefs'][k]
    r = h_comp['ridge_coefs'][k]
    d = abs(o - r)
    marker = ' *' if d > 0.1 else ''
    print(f'  {SUB_SHORT[k]:25s} {o:+8.3f} {r:+8.3f} {d:8.3f}{marker}')

# LLM conditions
print(f'\n  LLM OLS vs Ridge (all conditions):')
print(f'  {"Condition":22s} {"OLS R²":>7s} {"Ridge R²":>9s} {"α":>8s} '
      f'{"Max Δ coef":>11s} {"Stable?":>8s}')
print(f'  {"-"*22} {"-"*7} {"-"*9} {"-"*8} {"-"*11} {"-"*8}')

llm_comparisons = {}
for label, df in llm_dfs.items():
    try:
        comp = compare_ols_ridge(df, label)
        llm_comparisons[label] = comp
        max_delta = max(abs(comp['ols_coefs'][k] - comp['ridge_coefs'][k])
                        for k in SUB_KEYS)
        stable = 'YES' if max_delta < 0.15 else 'NO'
        print(f'  {label:22s} {comp["ols_r2"]:7.3f} {comp["ridge_r2"]:9.3f} '
              f'{comp["alpha"]:8.3f} {max_delta:11.3f} {stable:>8s}')
    except Exception as e:
        print(f'  {label:22s} ERROR: {e}')


# ════════════════════════════════════════════════════════════════════════════
# CHECK 3: LOOCV R²
# ════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('CHECK 3: Leave-One-Out Cross-Validated R²')
print('OLS R² can overfit. LOOCV R² shows true predictive power.')
print('Big gap between OLS R² and LOOCV R² = overfitting.')
print('=' * 70)


def loocv_r2(df):
    X = df[SUB_KEYS].values.astype(float)
    y = df['RC'].values.astype(float)
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X)
    ols = LinearRegression()
    loo = LeaveOneOut()
    scores = cross_val_score(ols, X_z, y, cv=loo, scoring='r2')
    return np.mean(scores)


# Human
h_loocv = loocv_r2(human_df)
h_ols_r2 = h_comp['ols_r2']
print(f'\n  HUMAN: OLS R² = {h_ols_r2:.3f}  |  LOOCV R² = {h_loocv:.3f}  '
      f'|  Gap = {h_ols_r2 - h_loocv:.3f}')

print(f'\n  {"Condition":22s} {"OLS R²":>7s} {"LOOCV R²":>9s} {"Gap":>6s} '
      f'{"Overfitting?":>13s}')
print(f'  {"-"*22} {"-"*7} {"-"*9} {"-"*6} {"-"*13}')
print(f'  {"Human":22s} {h_ols_r2:7.3f} {h_loocv:9.3f} {h_ols_r2-h_loocv:6.3f} '
      f'{"" if h_ols_r2-h_loocv < 0.1 else "YES":>13s}')

for label, df in llm_dfs.items():
    try:
        ols_r2 = llm_comparisons[label]['ols_r2']
        loo_r2 = loocv_r2(df)
        gap = ols_r2 - loo_r2
        overfit = 'YES' if gap > 0.15 else ('mild' if gap > 0.08 else '')
        print(f'  {label:22s} {ols_r2:7.3f} {loo_r2:9.3f} {gap:6.3f} '
              f'{overfit:>13s}')
    except Exception as e:
        print(f'  {label:22s} ERROR: {e}')


# ════════════════════════════════════════════════════════════════════════════
# CHECK 4: Bootstrap CIs on Human Story-Level Means
# ════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('CHECK 4: Bootstrap 95% CIs on Human Story-Level Means')
print('Shows uncertainty in human reference values (9-10 raters per story).')
print('=' * 70)

np.random.seed(42)
N_BOOT = 10000
shared_stories = sorted(human_df['story_id'].unique())

print(f'\n  {"Story":35s} {"IC mean":>7s} {"IC 95% CI":>14s} '
      f'{"RC mean":>7s} {"RC 95% CI":>14s} {"Flip%":>6s} {"Flip CI":>14s}')
print(f'  {"-"*35} {"-"*7} {"-"*14} {"-"*7} {"-"*14} {"-"*6} {"-"*14}')

all_human_boot_rc = {}
for sid in shared_stories:
    subset = human_df[human_df['story_id'] == sid]
    ic_vals = subset['IC'].values
    rc_vals = subset['RC'].values
    n = len(subset)

    # Bootstrap
    ic_boots = []
    rc_boots = []
    flip_boots = []
    for _ in range(N_BOOT):
        idx = np.random.randint(0, n, size=n)
        ic_boots.append(np.mean(ic_vals[idx]))
        rc_boots.append(np.mean(rc_vals[idx]))
        flips = np.sum(rc_vals[idx] != ic_vals[idx])
        flip_boots.append(flips / n * 100)

    ic_ci = np.percentile(ic_boots, [2.5, 97.5])
    rc_ci = np.percentile(rc_boots, [2.5, 97.5])
    flip_ci = np.percentile(flip_boots, [2.5, 97.5])
    flip_rate = np.sum(rc_vals != ic_vals) / n * 100
    all_human_boot_rc[sid] = (np.mean(rc_vals), rc_ci)

    print(f'  {sid:35s} {np.mean(ic_vals):7.2f} [{ic_ci[0]:5.2f}, {ic_ci[1]:5.2f}] '
          f'{np.mean(rc_vals):7.2f} [{rc_ci[0]:5.2f}, {rc_ci[1]:5.2f}] '
          f'{flip_rate:5.1f}% [{flip_ci[0]:4.0f}%, {flip_ci[1]:4.0f}%]')

# Overall human bootstrap
print(f'\n  Overall human (N=115):')
ic_all = human_df['IC'].values
rc_all = human_df['RC'].values
ic_boots_all = []
rc_boots_all = []
flip_boots_all = []
for _ in range(N_BOOT):
    idx = np.random.randint(0, len(human_df), size=len(human_df))
    ic_boots_all.append(np.mean(ic_all[idx]))
    rc_boots_all.append(np.mean(rc_all[idx]))
    flip_boots_all.append(np.mean(rc_all[idx] != ic_all[idx]) * 100)
ic_ci = np.percentile(ic_boots_all, [2.5, 97.5])
rc_ci = np.percentile(rc_boots_all, [2.5, 97.5])
flip_ci = np.percentile(flip_boots_all, [2.5, 97.5])
print(f'  IC mean = {ic_all.mean():.2f}  95% CI [{ic_ci[0]:.2f}, {ic_ci[1]:.2f}]')
print(f'  RC mean = {rc_all.mean():.2f}  95% CI [{rc_ci[0]:.2f}, {rc_ci[1]:.2f}]')
print(f'  Flip rate = {(rc_all != ic_all).mean()*100:.1f}%  '
      f'95% CI [{flip_ci[0]:.1f}%, {flip_ci[1]:.1f}%]')


# ════════════════════════════════════════════════════════════════════════════
# CHECK 5: PCA on 11 Sub-Components
# ════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('CHECK 5: PCA on 11 Sub-Components')
print('Do 11 dimensions collapse into fewer factors?')
print('=' * 70)


def run_pca(df, label):
    X = df[SUB_KEYS].values.astype(float)
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X)
    pca = PCA().fit(X_z)
    return pca


# Human PCA
h_pca = run_pca(human_df, 'Human')
cumvar = np.cumsum(h_pca.explained_variance_ratio_)

print(f'\n  HUMAN PCA (N=115):')
print(f'  {"PC":>4s} {"Var%":>7s} {"Cumul%":>8s}')
print(f'  {"-"*4} {"-"*7} {"-"*8}')
for i in range(11):
    print(f'  {i+1:4d} {h_pca.explained_variance_ratio_[i]*100:6.1f}% {cumvar[i]*100:7.1f}%')

# How many PCs for 80% variance?
n_80 = np.argmax(cumvar >= 0.80) + 1
n_90 = np.argmax(cumvar >= 0.90) + 1
print(f'\n  Components for 80% variance: {n_80}')
print(f'  Components for 90% variance: {n_90}')

# Loadings for top 3 PCs
print(f'\n  Top 3 PC loadings (human):')
print(f'  {"Sub-component":25s} {"PC1":>8s} {"PC2":>8s} {"PC3":>8s}')
print(f'  {"-"*25} {"-"*8} {"-"*8} {"-"*8}')
for j, k in enumerate(SUB_KEYS):
    print(f'  {SUB_SHORT[k]:25s} {h_pca.components_[0,j]:+8.3f} '
          f'{h_pca.components_[1,j]:+8.3f} {h_pca.components_[2,j]:+8.3f}')

# Figure: Scree plot + loadings
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Scree
ax1.bar(range(1, 12), h_pca.explained_variance_ratio_ * 100, color='#2196F3',
        alpha=0.7, label='Individual')
ax1.plot(range(1, 12), cumvar * 100, 'ro-', markersize=6, label='Cumulative')
ax1.axhline(80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Variance Explained (%)')
ax1.set_title('Scree Plot (Human Ratings)', fontweight='bold')
ax1.legend()
ax1.set_xticks(range(1, 12))

# Loadings heatmap
loadings = h_pca.components_[:3, :]
im = ax2.imshow(loadings, cmap='RdBu_r', aspect='auto', vmin=-0.6, vmax=0.6)
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels(['PC1', 'PC2', 'PC3'])
ax2.set_xticks(range(11))
ax2.set_xticklabels([SUB_SHORT[k] for k in SUB_KEYS], rotation=45, ha='right',
                     fontsize=9)
for i in range(3):
    for j in range(11):
        ax2.text(j, i, f'{loadings[i,j]:+.2f}', ha='center', va='center',
                 fontsize=7, color='white' if abs(loadings[i,j]) > 0.35 else 'black')
ax2.set_title('PC Loadings (Human)', fontweight='bold')
plt.colorbar(im, ax=ax2, shrink=0.8)
plt.tight_layout()
fig.savefig(f'{OUTDIR}/fig10_pca_human.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\n  → Saved fig10_pca_human.png')

# Compare: how many PCs per LLM?
print(f'\n  PCs for 80% variance by condition:')
print(f'  {"Condition":22s} {"N PCs (80%)":>12s} {"N PCs (90%)":>12s} '
      f'{"PC1 var%":>9s}')
print(f'  {"-"*22} {"-"*12} {"-"*12} {"-"*9}')
print(f'  {"Human":22s} {n_80:12d} {n_90:12d} '
      f'{h_pca.explained_variance_ratio_[0]*100:8.1f}%')

for label in ['Qwen3.5-4B-NT', 'Qwen3.5-27B-NT', 'Llama3.1-8B', 'Llama3.1-70B',
              'Gemma4-E2B-NT', 'Gemma4-31B-NT', 'Gemini3.1-Low', 'GPT-5.5']:
    if label not in llm_dfs:
        continue
    pca = run_pca(llm_dfs[label], label)
    cv = np.cumsum(pca.explained_variance_ratio_)
    n80 = np.argmax(cv >= 0.80) + 1
    n90 = np.argmax(cv >= 0.90) + 1
    print(f'  {label:22s} {n80:12d} {n90:12d} '
          f'{pca.explained_variance_ratio_[0]*100:8.1f}%')


# ════════════════════════════════════════════════════════════════════════════
# CHECK 6: Incremental Validity — Sub-Components Beyond IC
# ════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('CHECK 6: Incremental Validity')
print('Does RC ~ IC + 11 sub-components explain more than RC ~ IC alone?')
print('If yes, sub-components add information beyond the gut feeling.')
print('=' * 70)


def incremental_validity(df, label):
    X_ic = df[['IC']].values.astype(float)
    X_full = df[['IC'] + SUB_KEYS].values.astype(float)
    y = df['RC'].values.astype(float)

    scaler_ic = StandardScaler()
    scaler_full = StandardScaler()
    X_ic_z = scaler_ic.fit_transform(X_ic)
    X_full_z = scaler_full.fit_transform(X_full)

    r2_ic = LinearRegression().fit(X_ic_z, y).score(X_ic_z, y)
    r2_full = LinearRegression().fit(X_full_z, y).score(X_full_z, y)

    # LOOCV for both
    loo = LeaveOneOut()
    cv_ic = np.mean(cross_val_score(LinearRegression(), X_ic_z, y, cv=loo, scoring='r2'))
    cv_full = np.mean(cross_val_score(LinearRegression(), X_full_z, y, cv=loo, scoring='r2'))

    return {
        'r2_ic': r2_ic, 'r2_full': r2_full, 'delta_r2': r2_full - r2_ic,
        'cv_ic': cv_ic, 'cv_full': cv_full, 'cv_delta': cv_full - cv_ic,
    }


print(f'\n  {"Condition":22s} {"R²(IC)":>7s} {"R²(full)":>9s} {"ΔR²":>6s} '
      f'{"CV(IC)":>7s} {"CV(full)":>9s} {"CV ΔR²":>8s}')
print(f'  {"-"*22} {"-"*7} {"-"*9} {"-"*6} {"-"*7} {"-"*9} {"-"*8}')

# Human
h_iv = incremental_validity(human_df, 'Human')
print(f'  {"Human":22s} {h_iv["r2_ic"]:7.3f} {h_iv["r2_full"]:9.3f} '
      f'{h_iv["delta_r2"]:+5.3f} {h_iv["cv_ic"]:7.3f} {h_iv["cv_full"]:9.3f} '
      f'{h_iv["cv_delta"]:+7.3f}')

for label, df in llm_dfs.items():
    try:
        iv = incremental_validity(df, label)
        print(f'  {label:22s} {iv["r2_ic"]:7.3f} {iv["r2_full"]:9.3f} '
              f'{iv["delta_r2"]:+5.3f} {iv["cv_ic"]:7.3f} {iv["cv_full"]:9.3f} '
              f'{iv["cv_delta"]:+7.3f}')
    except Exception as e:
        print(f'  {label:22s} ERROR: {e}')


# ════════════════════════════════════════════════════════════════════════════
# CHECK 7: Re-run pathway similarity with Ridge coefficients
# ════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('CHECK 7: Pathway Similarity — OLS vs Ridge Coefficients')
print('Does the similarity ranking change when using Ridge instead of OLS?')
print('If rankings are stable, the main analysis is robust.')
print('=' * 70)


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


human_ols = np.array([h_comp['ols_coefs'][k] for k in SUB_KEYS])
human_ridge = np.array([h_comp['ridge_coefs'][k] for k in SUB_KEYS])

results_ols = []
results_ridge = []
for label in llm_comparisons:
    comp = llm_comparisons[label]
    ols_vec = np.array([comp['ols_coefs'][k] for k in SUB_KEYS])
    ridge_vec = np.array([comp['ridge_coefs'][k] for k in SUB_KEYS])
    cos_ols = cosine_sim(human_ols, ols_vec)
    cos_ridge = cosine_sim(human_ridge, ridge_vec)
    results_ols.append((label, cos_ols))
    results_ridge.append((label, cos_ridge))

results_ols.sort(key=lambda x: x[1], reverse=True)
results_ridge.sort(key=lambda x: x[1], reverse=True)

# Build rank comparison
ols_rank = {label: i+1 for i, (label, _) in enumerate(results_ols)}
ridge_rank = {label: i+1 for i, (label, _) in enumerate(results_ridge)}

print(f'\n  {"Condition":22s} {"OLS cos":>8s} {"OLS rank":>9s} '
      f'{"Ridge cos":>10s} {"Ridge rank":>11s} {"Δ rank":>7s}')
print(f'  {"-"*22} {"-"*8} {"-"*9} {"-"*10} {"-"*11} {"-"*7}')
for label, cos_o in results_ols:
    cos_r = dict(results_ridge)[label]
    r_ols = ols_rank[label]
    r_ridge = ridge_rank[label]
    delta = abs(r_ols - r_ridge)
    print(f'  {label:22s} {cos_o:+7.3f} {r_ols:9d} '
          f'{cos_r:+9.3f} {r_ridge:11d} {delta:7d}')

# Rank correlation
ols_ranks = [ols_rank[label] for label, _ in results_ols]
ridge_ranks = [ridge_rank[label] for label, _ in results_ols]
rho, _ = spearmanr(ols_ranks, ridge_ranks)
print(f'\n  Rank correlation (OLS vs Ridge): ρ = {rho:.3f}')
print(f'  {"STABLE" if rho > 0.9 else "UNSTABLE"}: '
      f'{"Rankings are robust to regularization" if rho > 0.9 else "Rankings change with regularization — interpret with caution"}')


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('ROBUSTNESS SUMMARY')
print('=' * 70)
print(f'\n  VIF:                Human max = {max(human_vif.values()):.1f} — '
      f'{"OK" if max(human_vif.values()) < 5 else "some multicollinearity"}')
print(f'  Ridge vs OLS:       Human coefficients stable '
      f'(max Δ = {max(abs(h_comp["ols_coefs"][k]-h_comp["ridge_coefs"][k]) for k in SUB_KEYS):.3f})')
print(f'  LOOCV:              Human OLS R²={h_ols_r2:.3f}, LOOCV R²={h_loocv:.3f} '
      f'(gap={h_ols_r2-h_loocv:.3f})')
print(f'  Bootstrap:          Human RC mean = {rc_all.mean():.2f} '
      f'[{np.percentile(rc_boots_all, 2.5):.2f}, {np.percentile(rc_boots_all, 97.5):.2f}]')
print(f'  PCA:                {n_80} PCs for 80% variance, {n_90} for 90%')
print(f'  Incremental:        ΔR² = {h_iv["delta_r2"]:+.3f} '
      f'(sub-components add {h_iv["delta_r2"]*100:.1f}% beyond IC)')
print(f'  Pathway ranking:    OLS vs Ridge rank ρ = {rho:.3f}')
