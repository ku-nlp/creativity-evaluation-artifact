"""
Cross-cutting analysis: Human vs LLM creativity judgment patterns.

Analyses:
  1. RC ~ 11 sub-components regression (human + each LLM condition)
  2. Score distributions (IC and RC) — human vs all LLMs
  3. Flip rate comparison — overall human vs each LLM
  4. Story-level ranking on 12 shared stories
  5. Pathway similarity — which LLM best matches human regression coefficients

Outputs figures to study2_llm/analysis_figures/
Prints all numbers to stdout.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ──────────────────────────────────────────────────────────────────

OUTDIR = 'study2_llm/analysis_figures'
os.makedirs(OUTDIR, exist_ok=True)

# Human sub-component column names → standardized names
HUMAN_SUB_MAP = {
    'R_Emotion': 'emotional_impact',
    'A_Topic': 'topic_fidelity',
    'N_Vocab': 'vocabulary_freshness',
    'N_Plot': 'plot_uniqueness',
    'N_Surprise': 'surprise',
    'R_Empathy': 'empathy',
    'R_Thought': 'thought_provocation',
    'V_Engagement': 'engagement',
    'V_Style': 'stylistic_quality',
    'V_Logic': 'logical_coherence',
    'A_Tone': 'tone_fidelity',
}

SUB_KEYS = [
    'emotional_impact', 'topic_fidelity', 'vocabulary_freshness',
    'plot_uniqueness', 'surprise', 'empathy', 'thought_provocation',
    'engagement', 'stylistic_quality', 'logical_coherence', 'tone_fidelity'
]

# Short labels for plots
SUB_SHORT = {
    'emotional_impact': 'Emotion',
    'topic_fidelity': 'Topic Fid.',
    'vocabulary_freshness': 'Vocab',
    'plot_uniqueness': 'Plot',
    'surprise': 'Surprise',
    'empathy': 'Empathy',
    'thought_provocation': 'Thought',
    'engagement': 'Engagement',
    'stylistic_quality': 'Style',
    'logical_coherence': 'Logic',
    'tone_fidelity': 'Tone Fid.',
}

# Human topic mapping to LLM story ID prefix
TOPIC_TO_PREFIX = {
    'An advanced AI initiates its own permanent shutdown sequence': 'ai_shutdown',
    'A lone worker in a Japanese convenience store at 3:00 AM. The city is silent.': 'midnight_store',
    'A professional thief attempting to crack a high-security safe in a dark room. High tension.': 'the_heist',
}

# LLM conditions to analyze (label → file path)
LLM_CONDITIONS = {
    # Qwen 3.5
    'Qwen3.5-4B-NT':    'study2_llm/data/raw_json/results_qwen35_4b_CLEAN_nothink.json',
    'Qwen3.5-4B-T':     'study2_llm/data/raw_json/results_qwen35_4b_CLEAN_think.json',
    'Qwen3.5-9B-NT':    'study2_llm/data/raw_json/results_qwen35_9b_CLEAN_nothink.json',
    'Qwen3.5-9B-T':     'study2_llm/data/raw_json/results_qwen35_9b_CLEAN_think.json',
    'Qwen3.5-27B-NT':   'study2_llm/data/raw_json/results_qwen35_27b_CLEAN_nothink.json',
    'Qwen3.5-27B-T':    'study2_llm/data/raw_json/results_qwen35_27b_CLEAN_think.json',
    'Qwen3.5-122B-NT':  'study2_llm/data/raw_json/results_qwen35_122b_judge_gemini_nothink.json',
    'Qwen3.5-122B-T':   'study2_llm/data/raw_json/results_qwen35_122b_judge_gemini_think.json',
    # Llama 3.1
    'Llama3.1-8B':      'study2_llm/data/raw_json/results_llama31_8b_judge_gemini_nothink.json',
    'Llama3.1-70B':     'study2_llm/data/raw_json/results_llama31_70b_judge_gemini_nothink.json',
    # Gemma 4
    'Gemma4-E2B-NT':    'study2_llm/data/raw_json/results_gemma4_e2b_nothink.json',
    'Gemma4-E2B-T':     'study2_llm/data/raw_json/results_gemma4_e2b_think.json',
    'Gemma4-E4B-NT':    'study2_llm/data/raw_json/results_gemma4_e4b_nothink.json',
    'Gemma4-E4B-T':     'study2_llm/data/raw_json/results_gemma4_e4b_think.json',
    'Gemma4-26B-NT':    'study2_llm/data/raw_json/results_gemma4_26b_nothink.json',
    'Gemma4-26B-T':     'study2_llm/data/raw_json/results_gemma4_26b_think.json',
    'Gemma4-31B-NT':    'study2_llm/data/raw_json/results_gemma4_31b_nothink.json',
    'Gemma4-31B-T':     'study2_llm/data/raw_json/results_gemma4_31b_think.json',
    # Gemini (proprietary)
    'Gemini3.1-Low':    'study2_llm/data/raw_json/results_gemini_31_pro_low.json',
    'Gemini3.1-Med':    'study2_llm/data/raw_json/results_gemini_31_pro_medium.json',
    'Gemini3.1-High':   'study2_llm/data/raw_json/results_gemini_31_pro_high.json',
    'Gemini3-Low':      'study2_llm/data/raw_json/results_gemini_3_pro_low.json',
    'Gemini3-Med':      'study2_llm/data/raw_json/results_gemini_3_pro_medium.json',
    'Gemini3-High':     'study2_llm/data/raw_json/results_gemini_3_pro_high.json',
    # GPT (proprietary)
    'GPT-5.5':          'study2_llm/data/raw_json/results_gpt55_CLEAN_high.json',
}

# ── Load data ───────────────────────────────────────────────────────────────

def load_human():
    """Load human ratings, return DataFrame with standardized column names."""
    df = pd.read_csv('study1_human/data/human_ratings.csv')
    # Rename sub-components
    rename = {v_human: v_std for v_human, v_std in HUMAN_SUB_MAP.items()}
    rename['O_Initial_Creativity'] = 'IC'
    rename['O_Final_Creativity'] = 'RC'
    rename['O_Enjoyment'] = 'enjoyment'
    df = df.rename(columns=rename)
    # Build story_id to match LLM format
    df['story_id'] = df.apply(
        lambda r: TOPIC_TO_PREFIX[r['TOPIC']] + '_' + r['TONE'].lower(), axis=1
    )
    return df


def load_llm(path):
    """Load one LLM condition, return list of dicts with standardized keys."""
    with open(path) as f:
        data = json.load(f)
    rows = []
    for r in data['results']:
        if not r.get('parse_ok', True):
            continue
        row = {
            'story_id': r['story_id'],
            'IC': r['scores']['initial_creativity'],
            'RC': r['scores']['reflective_creativity'],
            'enjoyment': r['scores'].get('enjoyment', None),
        }
        for k in SUB_KEYS:
            row[k] = r['scores']['sub_components'][k]
        rows.append(row)
    return pd.DataFrame(rows)


# ── Analysis functions ──────────────────────────────────────────────────────

def run_regression(df, label):
    """OLS: RC ~ 11 sub-components. Returns dict of coefficients and R²."""
    X = df[SUB_KEYS].values.astype(float)
    y = df['RC'].values.astype(float)
    # Standardize X for comparable coefficients
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    # Avoid div by zero for zero-variance columns
    X_std[X_std == 0] = 1.0
    X_z = (X - X_mean) / X_std
    model = LinearRegression().fit(X_z, y)
    r2 = model.score(X_z, y)
    coefs = dict(zip(SUB_KEYS, model.coef_))
    return {'label': label, 'r2': r2, 'coefs': coefs, 'intercept': model.intercept_}


def compute_flip_stats(df):
    """Compute flip rate, direction counts, avg shift."""
    flips = df['RC'] - df['IC']
    n = len(df)
    up = int((flips > 0).sum())
    down = int((flips < 0).sum())
    same = int((flips == 0).sum())
    rate = (up + down) / n * 100
    avg_shift = flips.mean()
    return {
        'n': n, 'up': up, 'down': down, 'same': same,
        'rate': rate, 'avg_shift': avg_shift
    }


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ── Main ────────────────────────────────────────────────────────────────────

print('=' * 70)
print('CROSS-ANALYSIS: Human vs LLM Creativity Judgment Patterns')
print('=' * 70)

# Load human
human_df = load_human()
print(f'\nHuman data: {len(human_df)} ratings, '
      f'{human_df["story_id"].nunique()} stories')

# Load all LLM conditions
llm_dfs = {}
for label, path in LLM_CONDITIONS.items():
    if os.path.exists(path):
        llm_dfs[label] = load_llm(path)
    else:
        print(f'  WARNING: {path} not found, skipping {label}')

print(f'LLM conditions loaded: {len(llm_dfs)}')

# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Regression — RC ~ 11 sub-components
# ════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('ANALYSIS 1: RC ~ 11 Sub-Components (Standardized OLS)')
print('What drives the final creativity score for each rater?')
print('Coefficients are standardized (z-scored X) so they are comparable.')
print('=' * 70)

# Human regression
human_reg = run_regression(human_df, 'Human')

# LLM regressions
llm_regs = {}
for label, df in llm_dfs.items():
    llm_regs[label] = run_regression(df, label)

# Print human baseline
print(f'\n{"HUMAN BASELINE":}')
print(f'  R² = {human_reg["r2"]:.3f}  (N = {len(human_df)})')
print(f'  {"Sub-component":25s} {"Coef":>8s}')
print(f'  {"-"*25} {"-"*8}')
for k in sorted(SUB_KEYS, key=lambda x: abs(human_reg['coefs'][x]), reverse=True):
    c = human_reg['coefs'][k]
    marker = ' ***' if abs(c) > 0.15 else ''
    print(f'  {SUB_SHORT[k]:25s} {c:+8.3f}{marker}')

# Print all LLM regressions
print(f'\n{"LLM CONDITIONS":}')
print(f'  {"Condition":22s} {"R²":>6s} | Top 3 predictors of RC')
print(f'  {"-"*22} {"-"*6} | {"-"*45}')
for label in llm_regs:
    reg = llm_regs[label]
    top3 = sorted(SUB_KEYS, key=lambda x: abs(reg['coefs'][x]), reverse=True)[:3]
    top3_str = ', '.join(f'{SUB_SHORT[k]}({reg["coefs"][k]:+.2f})' for k in top3)
    print(f'  {label:22s} {reg["r2"]:6.3f} | {top3_str}')

# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Score Distributions
# ════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('ANALYSIS 2: Score Distributions (IC and RC)')
print('Human: 115 ratings (inter-rater + inter-story variance)')
print('LLM: 80 ratings each (inter-story variance only)')
print('=' * 70)

print(f'\n  {"Condition":22s} {"IC mean":>7s} {"IC SD":>6s} {"IC%@7":>6s} | '
      f'{"RC mean":>7s} {"RC SD":>6s} {"RC%@7":>6s}')
print(f'  {"-"*22} {"-"*7} {"-"*6} {"-"*6} | {"-"*7} {"-"*6} {"-"*6}')

# Human
h_ic = human_df['IC'].values
h_rc = human_df['RC'].values
print(f'  {"Human":22s} {h_ic.mean():7.2f} {h_ic.std():6.2f} '
      f'{(h_ic==7).mean()*100:5.1f}% | '
      f'{h_rc.mean():7.2f} {h_rc.std():6.2f} '
      f'{(h_rc==7).mean()*100:5.1f}%')

# LLMs
for label, df in llm_dfs.items():
    ic = df['IC'].values
    rc = df['RC'].values
    print(f'  {label:22s} {ic.mean():7.2f} {ic.std():6.2f} '
          f'{(ic==7).mean()*100:5.1f}% | '
          f'{rc.mean():7.2f} {rc.std():6.2f} '
          f'{(rc==7).mean()*100:5.1f}%')

# ── Figure: Ridge plot of IC and RC distributions ──

# Select representative conditions for the figure (not all 25)
PLOT_CONDITIONS = [
    'Qwen3.5-4B-NT', 'Qwen3.5-4B-T',
    'Qwen3.5-27B-NT', 'Qwen3.5-122B-NT',
    'Llama3.1-8B', 'Llama3.1-70B',
    'Gemma4-E2B-NT', 'Gemma4-E4B-NT', 'Gemma4-E4B-T',
    'Gemma4-31B-NT',
    'Gemini3.1-Low', 'Gemini3.1-High',
    'Gemini3-Low', 'Gemini3-High',
    'GPT-5.5',
]

fig, axes = plt.subplots(len(PLOT_CONDITIONS) + 1, 2, figsize=(14, 28),
                         sharex=True)
fig.suptitle('Score Distributions: Human vs LLM Conditions',
             fontsize=16, fontweight='bold', y=0.995)
axes[0, 0].set_title('Initial Creativity (IC)', fontsize=13, fontweight='bold')
axes[0, 1].set_title('Reflective Creativity (RC)', fontsize=13, fontweight='bold')

bins = np.arange(0.5, 8.5, 1)  # 1-7

# Human row
for col, (vals, label_col) in enumerate([(h_ic, 'IC'), (h_rc, 'RC')]):
    ax = axes[0, col]
    counts = [np.sum(vals == v) for v in range(1, 8)]
    pcts = [c / len(vals) * 100 for c in counts]
    bars = ax.bar(range(1, 8), pcts, color='#2196F3', alpha=0.8, edgecolor='white')
    ax.set_ylabel('Human\n(N=115)', fontsize=10, fontweight='bold', rotation=0,
                  labelpad=70, va='center')
    ax.set_ylim(0, 100)
    ax.set_xlim(0.2, 7.8)
    ax.axhline(y=0, color='black', linewidth=0.5)
    for bar, pct in zip(bars, pcts):
        if pct > 3:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{pct:.0f}%', ha='center', va='bottom', fontsize=7)

# LLM rows
for i, cond in enumerate(PLOT_CONDITIONS):
    if cond not in llm_dfs:
        continue
    df = llm_dfs[cond]
    for col, score_col in enumerate(['IC', 'RC']):
        ax = axes[i + 1, col]
        vals = df[score_col].values
        counts = [np.sum(vals == v) for v in range(1, 8)]
        pcts = [c / len(vals) * 100 for c in counts]
        color = '#FF9800' if 'T' in cond.split('-')[-1] else '#4CAF50'
        if 'Gemini' in cond or 'GPT' in cond:
            color = '#9C27B0'
        bars = ax.bar(range(1, 8), pcts, color=color, alpha=0.8, edgecolor='white')
        ax.set_ylabel(f'{cond}', fontsize=8, fontweight='bold', rotation=0,
                      labelpad=70, va='center')
        ax.set_ylim(0, 100)
        ax.axhline(y=0, color='black', linewidth=0.5)
        for bar, pct in zip(bars, pcts):
            if pct > 3:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{pct:.0f}%', ha='center', va='bottom', fontsize=7)

axes[-1, 0].set_xlabel('Score (1-7)', fontsize=11)
axes[-1, 1].set_xlabel('Score (1-7)', fontsize=11)
plt.tight_layout(rect=[0.12, 0, 1, 0.99])
fig.savefig(f'{OUTDIR}/fig1_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\n  → Saved {OUTDIR}/fig1_distributions.png')

# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Flip Rate Comparison
# ════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('ANALYSIS 3: Revision (Flip) Rate — Human vs LLM')
print('Human flip = (RC ≠ IC). Direction = up if RC > IC, down if RC < IC.')
print('=' * 70)

human_flips = compute_flip_stats(human_df)
print(f'\n  {"Condition":22s} {"Rate":>6s} {"Up":>4s} {"Down":>5s} {"Same":>5s} '
      f'{"Avg Δ":>7s} {"N":>4s}')
print(f'  {"-"*22} {"-"*6} {"-"*4} {"-"*5} {"-"*5} {"-"*7} {"-"*4}')
print(f'  {"Human":22s} {human_flips["rate"]:5.1f}% '
      f'{human_flips["up"]:4d} {human_flips["down"]:5d} '
      f'{human_flips["same"]:5d} {human_flips["avg_shift"]:+6.2f} '
      f'{human_flips["n"]:4d}')

llm_flip_stats = {}
for label, df in llm_dfs.items():
    fs = compute_flip_stats(df)
    llm_flip_stats[label] = fs
    print(f'  {label:22s} {fs["rate"]:5.1f}% '
          f'{fs["up"]:4d} {fs["down"]:5d} {fs["same"]:5d} '
          f'{fs["avg_shift"]:+6.2f} {fs["n"]:4d}')

# ── Figure: Flip rate bar chart ──

fig, ax = plt.subplots(figsize=(16, 7))
labels_plot = ['Human'] + list(llm_flip_stats.keys())
up_rates = [human_flips['up'] / human_flips['n'] * 100]
down_rates = [human_flips['down'] / human_flips['n'] * 100]
for label in llm_flip_stats:
    fs = llm_flip_stats[label]
    up_rates.append(fs['up'] / fs['n'] * 100)
    down_rates.append(fs['down'] / fs['n'] * 100)

x = np.arange(len(labels_plot))
bars_up = ax.bar(x, up_rates, 0.6, label='Upward revision', color='#4CAF50', alpha=0.85)
bars_down = ax.bar(x, [-d for d in down_rates], 0.6, label='Downward revision',
                   color='#F44336', alpha=0.85)

ax.axhline(0, color='black', linewidth=0.8)
# Mark human rate with a horizontal line
human_total = human_flips['rate']
ax.axhline(human_flips['up']/human_flips['n']*100, color='#2196F3',
           linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(-human_flips['down']/human_flips['n']*100, color='#2196F3',
           linestyle='--', alpha=0.5, linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(labels_plot, rotation=60, ha='right', fontsize=8)
ax.set_ylabel('Revision Rate (%)', fontsize=11)
ax.set_title('Revision Direction: Human vs All LLM Conditions', fontsize=14,
             fontweight='bold')
ax.legend(loc='upper right')
plt.tight_layout()
fig.savefig(f'{OUTDIR}/fig2_flip_rates.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\n  → Saved {OUTDIR}/fig2_flip_rates.png')

# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Story-Level Ranking on 12 Shared Stories
# ════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('ANALYSIS 4: Story Rankings on 12 Shared Stories')
print('Spearman ρ between human mean RC and LLM RC for same 12 stories.')
print('=' * 70)

shared_stories = sorted(human_df['story_id'].unique())
print(f'\n  Shared stories: {shared_stories}')

# Human mean RC per story
human_story_rc = human_df.groupby('story_id')['RC'].mean()
human_story_ic = human_df.groupby('story_id')['IC'].mean()

print(f'\n  Human mean RC per story:')
for sid in shared_stories:
    n_raters = len(human_df[human_df['story_id'] == sid])
    print(f'    {sid:35s}  IC={human_story_ic[sid]:.2f}  '
          f'RC={human_story_rc[sid]:.2f}  (n={n_raters})')

print(f'\n  {"Condition":22s} {"ρ (RC)":>8s} {"ρ (IC)":>8s} | '
      f'{"LLM RC range":>14s} | Interpretation')
print(f'  {"-"*22} {"-"*8} {"-"*8} | {"-"*14} | {"-"*30}')

story_corrs = {}
for label, df in llm_dfs.items():
    # Filter to 12 shared stories
    df12 = df[df['story_id'].isin(shared_stories)]
    if len(df12) < 12:
        continue
    llm_rc = df12.set_index('story_id').loc[shared_stories, 'RC']
    llm_ic = df12.set_index('story_id').loc[shared_stories, 'IC']
    h_rc = human_story_rc.loc[shared_stories]
    h_ic = human_story_ic.loc[shared_stories]

    rho_rc, p_rc = spearmanr(h_rc.values, llm_rc.values)
    rho_ic, p_ic = spearmanr(h_ic.values, llm_ic.values)

    rc_range = f'{llm_rc.min():.0f}-{llm_rc.max():.0f}'

    interp = ''
    if rho_rc > 0.7:
        interp = 'Strong agreement'
    elif rho_rc > 0.4:
        interp = 'Moderate agreement'
    elif rho_rc > 0.1:
        interp = 'Weak agreement'
    else:
        interp = 'No agreement'

    story_corrs[label] = rho_rc
    print(f'  {label:22s} {rho_rc:+7.3f}  {rho_ic:+7.3f}  | '
          f'{rc_range:>14s} | {interp}')

# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Pathway Similarity
# ════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('ANALYSIS 5: Pathway Similarity — Which LLM Matches Human Pattern?')
print('Cosine similarity between human and LLM regression coefficient vectors.')
print('1.0 = identical pathway, 0.0 = orthogonal, -1.0 = opposite.')
print('Also: Spearman ρ on coefficient rankings (do they weight the same')
print('sub-components as important, even if magnitudes differ?).')
print('=' * 70)

human_coef_vec = np.array([human_reg['coefs'][k] for k in SUB_KEYS])

results = []
for label, reg in llm_regs.items():
    llm_coef_vec = np.array([reg['coefs'][k] for k in SUB_KEYS])
    cos_sim = cosine_similarity(human_coef_vec, llm_coef_vec)
    rho, _ = spearmanr(human_coef_vec, llm_coef_vec)
    results.append({
        'label': label,
        'cos_sim': cos_sim,
        'rho': rho,
        'r2': reg['r2'],
        'story_rho': story_corrs.get(label, float('nan')),
    })

results.sort(key=lambda x: x['cos_sim'], reverse=True)

print(f'\n  {"Condition":22s} {"Cos Sim":>8s} {"Coef ρ":>8s} {"R²":>6s} '
      f'{"Story ρ":>8s}')
print(f'  {"-"*22} {"-"*8} {"-"*8} {"-"*6} {"-"*8}')
for r in results:
    story_rho_str = f'{r["story_rho"]:+7.3f}' if not np.isnan(r['story_rho']) else '    N/A'
    print(f'  {r["label"]:22s} {r["cos_sim"]:+7.3f}  {r["rho"]:+7.3f}  '
          f'{r["r2"]:5.3f}  {story_rho_str}')

# ── Figure: Coefficient heatmap ──

fig, ax = plt.subplots(figsize=(16, 12))

# Build matrix: rows = conditions (sorted by similarity), cols = sub-components
row_labels = ['Human'] + [r['label'] for r in results]
matrix = np.zeros((len(row_labels), len(SUB_KEYS)))
matrix[0, :] = human_coef_vec
for i, r in enumerate(results):
    reg = llm_regs[r['label']]
    matrix[i + 1, :] = [reg['coefs'][k] for k in SUB_KEYS]

col_labels = [SUB_SHORT[k] for k in SUB_KEYS]

im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-0.6, vmax=0.6)
ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=9)

# Add text annotations
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        color = 'white' if abs(val) > 0.35 else 'black'
        ax.text(j, i, f'{val:+.2f}', ha='center', va='center',
                fontsize=7, color=color)

# Highlight human row
ax.axhline(0.5, color='gold', linewidth=2)

# Add similarity scores on the right
for i, r in enumerate(results):
    ax.text(len(col_labels) + 0.3, i + 1,
            f'cos={r["cos_sim"]:+.2f}', fontsize=8, va='center',
            fontfamily='monospace')

ax.set_title('Sub-Component → RC Regression Coefficients (Standardized)\n'
             'Sorted by cosine similarity to human pathway',
             fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Standardized Coefficient', shrink=0.6)
plt.tight_layout()
fig.savefig(f'{OUTDIR}/fig3_coefficient_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\n  → Saved {OUTDIR}/fig3_coefficient_heatmap.png')

# ── Figure: Pathway similarity summary ──

fig, ax = plt.subplots(figsize=(14, 7))
labels_sim = [r['label'] for r in results]
cos_vals = [r['cos_sim'] for r in results]

colors = []
for label in labels_sim:
    if 'Qwen' in label:
        colors.append('#E91E63')
    elif 'Llama' in label:
        colors.append('#FF9800')
    elif 'Gemma' in label:
        colors.append('#4CAF50')
    elif 'Gemini' in label:
        colors.append('#9C27B0')
    elif 'GPT' in label:
        colors.append('#2196F3')
    else:
        colors.append('#607D8B')

bars = ax.barh(range(len(labels_sim)), cos_vals, color=colors, alpha=0.85,
               edgecolor='white')
ax.set_yticks(range(len(labels_sim)))
ax.set_yticklabels(labels_sim, fontsize=9)
ax.set_xlabel('Cosine Similarity to Human Pathway', fontsize=12)
ax.set_title('Which LLM Judges Creativity Most Like Humans?',
             fontsize=14, fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8)
ax.invert_yaxis()

# Add value labels
for i, (bar, val) in enumerate(zip(bars, cos_vals)):
    ax.text(val + 0.02 if val >= 0 else val - 0.02,
            i, f'{val:+.3f}', va='center',
            ha='left' if val >= 0 else 'right', fontsize=8)

# Legend
patches = [
    mpatches.Patch(color='#E91E63', label='Qwen 3.5'),
    mpatches.Patch(color='#FF9800', label='Llama 3.1'),
    mpatches.Patch(color='#4CAF50', label='Gemma 4'),
    mpatches.Patch(color='#9C27B0', label='Gemini'),
    mpatches.Patch(color='#2196F3', label='GPT'),
]
ax.legend(handles=patches, loc='lower right', fontsize=10)
plt.tight_layout()
fig.savefig(f'{OUTDIR}/fig4_pathway_similarity.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  → Saved {OUTDIR}/fig4_pathway_similarity.png')

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════

print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)

# Top 3 most human-like by each metric
print('\nMost human-like by pathway (cosine similarity to human RC regression):')
for i, r in enumerate(results[:5]):
    print(f'  {i+1}. {r["label"]:22s}  cos={r["cos_sim"]:+.3f}')

print('\nMost human-like by story ranking (Spearman ρ on 12 shared stories):')
story_ranked = sorted(
    [(l, v) for l, v in story_corrs.items()],
    key=lambda x: x[1], reverse=True
)
for i, (label, rho) in enumerate(story_ranked[:5]):
    print(f'  {i+1}. {label:22s}  ρ={rho:+.3f}')

print('\nClosest to human flip rate ({:.1f}%):'.format(human_flips['rate']))
flip_ranked = sorted(
    llm_flip_stats.items(),
    key=lambda x: abs(x[1]['rate'] - human_flips['rate'])
)
for i, (label, fs) in enumerate(flip_ranked[:5]):
    has_down = 'yes' if fs['down'] > 0 else 'no'
    print(f'  {i+1}. {label:22s}  rate={fs["rate"]:.1f}%  '
          f'(down={has_down})')

print('\nHuman reference:')
print(f'  IC mean = {h_ic.mean():.2f}, RC mean = {h_rc.mean():.2f}')
print(f'  Flip rate = {human_flips["rate"]:.1f}%  '
      f'(up={human_flips["up"]}, down={human_flips["down"]})')
print(f'  RC pathway: Vocab(+{human_reg["coefs"]["vocabulary_freshness"]:.2f}), '
      f'Engagement(+{human_reg["coefs"]["engagement"]:.2f}), '
      f'Thought(+{human_reg["coefs"]["thought_provocation"]:.2f})')

print('\n' + '=' * 70)
print('All figures saved to:', OUTDIR)
print('=' * 70)
