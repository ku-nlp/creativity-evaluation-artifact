"""
Distribution analysis: How do human ratings look vs LLM ratings?

Compares:
  1. Per-story spread: human rater variance vs LLM cross-model variance
  2. Overall score distributions side by side
  3. Rating diversity index (how much of the 1-7 scale is actually used)

Outputs figures to study2_llm/analysis_figures/
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUTDIR = 'study2_llm/analysis_figures'
os.makedirs(OUTDIR, exist_ok=True)

# ── Column mapping ──────────────────────────────────────────────────────────

HUMAN_SUB_MAP = {
    'R_Emotion': 'emotional_impact', 'A_Topic': 'topic_fidelity',
    'N_Vocab': 'vocabulary_freshness', 'N_Plot': 'plot_uniqueness',
    'N_Surprise': 'surprise', 'R_Empathy': 'empathy',
    'R_Thought': 'thought_provocation', 'V_Engagement': 'engagement',
    'V_Style': 'stylistic_quality', 'V_Logic': 'logical_coherence',
    'A_Tone': 'tone_fidelity',
}
SUB_KEYS = list(HUMAN_SUB_MAP.values())

TOPIC_TO_PREFIX = {
    'An advanced AI initiates its own permanent shutdown sequence': 'ai_shutdown',
    'A lone worker in a Japanese convenience store at 3:00 AM. The city is silent.': 'midnight_store',
    'A professional thief attempting to crack a high-security safe in a dark room. High tension.': 'the_heist',
}

# All LLM conditions grouped
LLM_FILES = {
    # Small open-source (≤9B)
    'Qwen 4B NT':     'study2_llm/data/raw_json/results_qwen35_4b_CLEAN_nothink.json',
    'Qwen 4B T':      'study2_llm/data/raw_json/results_qwen35_4b_CLEAN_think.json',
    'Qwen 9B NT':     'study2_llm/data/raw_json/results_qwen35_9b_CLEAN_nothink.json',
    'Qwen 9B T':      'study2_llm/data/raw_json/results_qwen35_9b_CLEAN_think.json',
    'Gemma E2B NT':   'study2_llm/data/raw_json/results_gemma4_e2b_nothink.json',
    'Gemma E2B T':    'study2_llm/data/raw_json/results_gemma4_e2b_think.json',
    'Gemma E4B NT':   'study2_llm/data/raw_json/results_gemma4_e4b_nothink.json',
    'Gemma E4B T':    'study2_llm/data/raw_json/results_gemma4_e4b_think.json',
    'Llama 8B':       'study2_llm/data/raw_json/results_llama31_8b_judge_gemini_nothink.json',
    # Large open-source (≥26B)
    'Qwen 27B NT':    'study2_llm/data/raw_json/results_qwen35_27b_CLEAN_nothink.json',
    'Qwen 27B T':     'study2_llm/data/raw_json/results_qwen35_27b_CLEAN_think.json',
    'Qwen 122B NT':   'study2_llm/data/raw_json/results_qwen35_122b_judge_gemini_nothink.json',
    'Qwen 122B T':    'study2_llm/data/raw_json/results_qwen35_122b_judge_gemini_think.json',
    'Gemma 26B NT':   'study2_llm/data/raw_json/results_gemma4_26b_nothink.json',
    'Gemma 26B T':    'study2_llm/data/raw_json/results_gemma4_26b_think.json',
    'Gemma 31B NT':   'study2_llm/data/raw_json/results_gemma4_31b_nothink.json',
    'Gemma 31B T':    'study2_llm/data/raw_json/results_gemma4_31b_think.json',
    'Llama 70B':      'study2_llm/data/raw_json/results_llama31_70b_judge_gemini_nothink.json',
    # Proprietary
    'Gemini 3.1 Low': 'study2_llm/data/raw_json/results_gemini_31_pro_low.json',
    'Gemini 3.1 Med': 'study2_llm/data/raw_json/results_gemini_31_pro_medium.json',
    'Gemini 3.1 Hi':  'study2_llm/data/raw_json/results_gemini_31_pro_high.json',
    'Gemini 3 Low':   'study2_llm/data/raw_json/results_gemini_3_pro_low.json',
    'Gemini 3 Med':   'study2_llm/data/raw_json/results_gemini_3_pro_medium.json',
    'Gemini 3 Hi':    'study2_llm/data/raw_json/results_gemini_3_pro_high.json',
    'GPT-5.5':        'study2_llm/data/raw_json/results_gpt55_CLEAN_high.json',
}

SIZE_GROUP = {}
for k in ['Qwen 4B NT','Qwen 4B T','Qwen 9B NT','Qwen 9B T',
          'Gemma E2B NT','Gemma E2B T','Gemma E4B NT','Gemma E4B T','Llama 8B']:
    SIZE_GROUP[k] = 'Small (≤9B)'
for k in ['Qwen 27B NT','Qwen 27B T','Qwen 122B NT','Qwen 122B T',
          'Gemma 26B NT','Gemma 26B T','Gemma 31B NT','Gemma 31B T','Llama 70B']:
    SIZE_GROUP[k] = 'Large (≥26B)'
for k in ['Gemini 3.1 Low','Gemini 3.1 Med','Gemini 3.1 Hi',
          'Gemini 3 Low','Gemini 3 Med','Gemini 3 Hi','GPT-5.5']:
    SIZE_GROUP[k] = 'Proprietary'


# ── Load data ───────────────────────────────────────────────────────────────

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

print(f'Human: {len(human_df)} ratings, {human_df["story_id"].nunique()} stories')
print(f'LLM conditions: {len(llm_dfs)}')

# 12 shared stories
shared_stories = sorted(human_df['story_id'].unique())


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Master distribution — Human vs every LLM (IC and RC histograms)
# ════════════════════════════════════════════════════════════════════════════

# Select key conditions for a clean figure
SELECTED = [
    'Qwen 4B NT', 'Qwen 9B NT', 'Qwen 27B NT',
    'Qwen 122B NT',
    'Llama 8B', 'Llama 70B',
    'Gemma E2B NT', 'Gemma E4B NT', 'Gemma 31B NT',
    'Gemini 3.1 Low', 'Gemini 3 Low',
    'GPT-5.5',
]
SELECTED = [s for s in SELECTED if s in llm_dfs]

n_rows = len(SELECTED) + 1
fig, axes = plt.subplots(n_rows, 2, figsize=(12, n_rows * 1.8 + 1),
                         sharex=True, sharey=True)
fig.suptitle('How Ratings Are Distributed: Human vs LLM (Nothink, 1 per family/size)',
             fontsize=14, fontweight='bold', y=1.0)
axes[0, 0].set_title('Initial Creativity (IC)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Reflective Creativity (RC)', fontsize=12, fontweight='bold')

def plot_hist(ax, vals, color, label, n_total):
    counts = [np.sum(vals == v) for v in range(1, 8)]
    pcts = [c / n_total * 100 for c in counts]
    bars = ax.bar(range(1, 8), pcts, color=color, alpha=0.85, edgecolor='white',
                  width=0.8)
    ax.set_ylabel(label, fontsize=8, fontweight='bold', rotation=0,
                  labelpad=80, va='center', ha='right')
    ax.set_ylim(0, 100)
    ax.set_xlim(0.2, 7.8)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=7)
    # Add percentage labels on bars
    for bar, pct in zip(bars, pcts):
        if pct > 5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f'{pct:.0f}%', ha='center', va='bottom', fontsize=7)
    # Add mean line
    mean_val = np.mean(vals)
    ax.axvline(mean_val, color='red', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.text(mean_val + 0.15, 85, f'μ={mean_val:.1f}', fontsize=7, color='red')

# Human
plot_hist(axes[0, 0], human_df['IC'].values, '#2196F3', f'Human\n(N={len(human_df)})', len(human_df))
plot_hist(axes[0, 1], human_df['RC'].values, '#2196F3', f'Human\n(N={len(human_df)})', len(human_df))
# Thick border for human row
for col in [0, 1]:
    for spine in axes[0, col].spines.values():
        spine.set_linewidth(2)
        spine.set_color('#2196F3')

# LLMs
colors_map = {
    'Small (≤9B)': '#4CAF50',
    'Large (≥26B)': '#FF9800',
    'Proprietary': '#9C27B0',
}
for i, cond in enumerate(SELECTED):
    df = llm_dfs[cond]
    grp = SIZE_GROUP[cond]
    color = colors_map[grp]
    n = len(df)
    row_label = f'{cond}\n(N={n})'
    plot_hist(axes[i+1, 0], df['IC'].values, color, row_label, n)
    plot_hist(axes[i+1, 1], df['RC'].values, color, row_label, n)

axes[-1, 0].set_xlabel('Score (1-7)', fontsize=11)
axes[-1, 1].set_xlabel('Score (1-7)', fontsize=11)
plt.tight_layout(rect=[0.14, 0, 1, 0.98])
fig.savefig(f'{OUTDIR}/fig5_master_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'→ Saved fig5_master_distributions.png')


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Per-story spread on 12 shared stories
# Human SD vs small-LLM spread vs large-LLM spread
# ════════════════════════════════════════════════════════════════════════════

small_models = [k for k in llm_dfs if SIZE_GROUP.get(k) == 'Small (≤9B)']
large_models = [k for k in llm_dfs if SIZE_GROUP.get(k) == 'Large (≥26B)']
prop_models = [k for k in llm_dfs if SIZE_GROUP.get(k) == 'Proprietary']

print('\n' + '='*70)
print('PER-STORY SPREAD ON 12 SHARED STORIES')
print('Human: SD across ~10 raters per story')
print('LLMs: SD across models within size group, for same story')
print('='*70)

print(f'\n  {"Story":35s} {"Human SD":>9s} {"Small SD":>9s} {"Large SD":>9s} {"Prop SD":>9s}')
print(f'  {"-"*35} {"-"*9} {"-"*9} {"-"*9} {"-"*9}')

human_sds = []
small_sds = []
large_sds = []
prop_sds = []

for sid in shared_stories:
    # Human: SD of RC across raters
    h_vals = human_df[human_df['story_id'] == sid]['RC'].values
    h_sd = np.std(h_vals, ddof=1) if len(h_vals) > 1 else 0

    # LLM: collect RC for this story across models in each group
    def get_group_scores(models):
        scores = []
        for m in models:
            df = llm_dfs[m]
            row = df[df['story_id'] == sid]
            if len(row) == 1:
                scores.append(row['RC'].values[0])
        return scores

    s_vals = get_group_scores(small_models)
    l_vals = get_group_scores(large_models)
    p_vals = get_group_scores(prop_models)

    s_sd = np.std(s_vals, ddof=1) if len(s_vals) > 1 else 0
    l_sd = np.std(l_vals, ddof=1) if len(l_vals) > 1 else 0
    p_sd = np.std(p_vals, ddof=1) if len(p_vals) > 1 else 0

    human_sds.append(h_sd)
    small_sds.append(s_sd)
    large_sds.append(l_sd)
    prop_sds.append(p_sd)

    print(f'  {sid:35s} {h_sd:9.2f} {s_sd:9.2f} {l_sd:9.2f} {p_sd:9.2f}')

print(f'\n  {"MEAN":35s} {np.mean(human_sds):9.2f} {np.mean(small_sds):9.2f} '
      f'{np.mean(large_sds):9.2f} {np.mean(prop_sds):9.2f}')

# Figure
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(shared_stories))
w = 0.2
bars1 = ax.bar(x - 1.5*w, human_sds, w, label=f'Human raters (N≈10)', color='#2196F3', alpha=0.85)
bars2 = ax.bar(x - 0.5*w, small_sds, w, label=f'Small LLMs ≤9B (N={len(small_models)})', color='#4CAF50', alpha=0.85)
bars3 = ax.bar(x + 0.5*w, large_sds, w, label=f'Large LLMs ≥26B (N={len(large_models)})', color='#FF9800', alpha=0.85)
bars4 = ax.bar(x + 1.5*w, prop_sds, w, label=f'Proprietary (N={len(prop_models)})', color='#9C27B0', alpha=0.85)

ax.set_xticks(x)
short_labels = [s.replace('ai_shutdown_', 'AI-').replace('midnight_store_', 'Store-').replace('the_heist_', 'Heist-')
                for s in shared_stories]
ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Standard Deviation of RC', fontsize=11)
ax.set_title('Rating Diversity Per Story: Human Raters vs LLM Groups\n'
             '(Higher SD = more disagreement = more diverse perspectives)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0, max(max(human_sds), max(small_sds), max(large_sds), max(prop_sds)) + 0.3)
plt.tight_layout()
fig.savefig(f'{OUTDIR}/fig6_per_story_spread.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'→ Saved fig6_per_story_spread.png')


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Scale usage — what fraction of 1-7 does each rater actually use?
# ════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('SCALE USAGE AND RATING DIVERSITY')
print('='*70)

def scale_stats(vals):
    """Compute how much of the 1-7 scale is used."""
    vals = np.array(vals, dtype=int)
    unique = len(np.unique(vals))
    rng = int(vals.max() - vals.min())
    sd = np.std(vals)
    # Shannon entropy of score distribution (max = log2(7) ≈ 2.81)
    counts = np.array([np.sum(vals == v) for v in range(1, 8)])
    probs = counts / counts.sum()
    h = entropy(probs, base=2)
    h_max = np.log2(7)
    h_norm = h / h_max  # 0 = all same score, 1 = perfectly uniform
    # Ceiling concentration
    ceil_pct = np.mean(vals == 7) * 100
    return {
        'unique_scores': unique, 'range': rng, 'sd': sd,
        'entropy': h, 'entropy_norm': h_norm, 'ceil_pct': ceil_pct,
        'mean': np.mean(vals),
    }

print(f'\n  {"Condition":22s} {"Mean":>5s} {"SD":>5s} {"Range":>6s} '
      f'{"Unique":>7s} {"Entropy":>8s} {"Norm H":>7s} {"Ceil%":>6s}')
print(f'  {"-"*22} {"-"*5} {"-"*5} {"-"*6} {"-"*7} {"-"*8} {"-"*7} {"-"*6}')

all_stats = {}

# Human IC
s = scale_stats(human_df['IC'].values)
all_stats['Human IC'] = s
print(f'  {"Human IC":22s} {s["mean"]:5.2f} {s["sd"]:5.2f} {s["range"]:6d} '
      f'{s["unique_scores"]:7d} {s["entropy"]:8.3f} {s["entropy_norm"]:7.3f} '
      f'{s["ceil_pct"]:5.1f}%')

# Human RC
s = scale_stats(human_df['RC'].values)
all_stats['Human RC'] = s
print(f'  {"Human RC":22s} {s["mean"]:5.2f} {s["sd"]:5.2f} {s["range"]:6d} '
      f'{s["unique_scores"]:7d} {s["entropy"]:8.3f} {s["entropy_norm"]:7.3f} '
      f'{s["ceil_pct"]:5.1f}%')

print(f'  {"-"*22} {"-"*5} {"-"*5} {"-"*6} {"-"*7} {"-"*8} {"-"*7} {"-"*6}')

for label, df in llm_dfs.items():
    s_ic = scale_stats(df['IC'].values)
    s_rc = scale_stats(df['RC'].values)
    all_stats[f'{label} IC'] = s_ic
    all_stats[f'{label} RC'] = s_rc
    print(f'  {label + " RC":22s} {s_rc["mean"]:5.2f} {s_rc["sd"]:5.2f} '
          f'{s_rc["range"]:6d} {s_rc["unique_scores"]:7d} '
          f'{s_rc["entropy"]:8.3f} {s_rc["entropy_norm"]:7.3f} '
          f'{s_rc["ceil_pct"]:5.1f}%')


# ── Figure: Entropy comparison ──

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: Normalized entropy (scale diversity)
labels_ent = ['Human']
vals_ent = [scale_stats(human_df['RC'].values)['entropy_norm']]
colors_ent = ['#2196F3']

for cond in SELECTED:
    df = llm_dfs[cond]
    s = scale_stats(df['RC'].values)
    labels_ent.append(cond)
    vals_ent.append(s['entropy_norm'])
    colors_ent.append(colors_map[SIZE_GROUP[cond]])

y_pos = np.arange(len(labels_ent))
ax1.barh(y_pos, vals_ent, color=colors_ent, alpha=0.85, edgecolor='white')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels_ent, fontsize=9)
ax1.set_xlabel('Normalized Shannon Entropy\n(0 = all same score, 1 = uniform across 1-7)', fontsize=10)
ax1.set_title('Scale Diversity (RC)', fontsize=13, fontweight='bold')
ax1.invert_yaxis()
ax1.axvline(vals_ent[0], color='#2196F3', linestyle='--', alpha=0.5, linewidth=1.5)
for i, v in enumerate(vals_ent):
    ax1.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=8)

# Right: Ceiling concentration
labels_ceil = ['Human']
vals_ceil = [scale_stats(human_df['RC'].values)['ceil_pct']]
colors_ceil = ['#2196F3']

for cond in SELECTED:
    df = llm_dfs[cond]
    s = scale_stats(df['RC'].values)
    labels_ceil.append(cond)
    vals_ceil.append(s['ceil_pct'])
    colors_ceil.append(colors_map[SIZE_GROUP[cond]])

ax2.barh(np.arange(len(labels_ceil)), vals_ceil, color=colors_ceil, alpha=0.85,
         edgecolor='white')
ax2.set_yticks(np.arange(len(labels_ceil)))
ax2.set_yticklabels(labels_ceil, fontsize=9)
ax2.set_xlabel('% of RC Scores at Ceiling (=7)', fontsize=10)
ax2.set_title('Ceiling Concentration (RC)', fontsize=13, fontweight='bold')
ax2.invert_yaxis()
ax2.axvline(vals_ceil[0], color='#2196F3', linestyle='--', alpha=0.5, linewidth=1.5)
for i, v in enumerate(vals_ceil):
    ax2.text(v + 0.5, i, f'{v:.0f}%', va='center', fontsize=8)

plt.tight_layout()
fig.savefig(f'{OUTDIR}/fig7_scale_diversity.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\n→ Saved fig7_scale_diversity.png')


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4: IC→RC shift distributions (the "what does reflection do?")
# ════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(len(SELECTED) + 1, 1, figsize=(10, n_rows * 1.6 + 1),
                         sharex=True, sharey=True)
fig.suptitle('What Does Reflection Do? Distribution of IC → RC Shifts',
             fontsize=14, fontweight='bold', y=1.0)

def plot_shift(ax, ic_vals, rc_vals, label, color):
    shifts = rc_vals - ic_vals
    bins = np.arange(-4.5, 5.5, 1)
    counts, _ = np.histogram(shifts, bins=bins)
    pcts = counts / len(shifts) * 100
    centers = np.arange(-4, 5)
    bar_colors = ['#F44336' if c < 0 else '#4CAF50' if c > 0 else '#9E9E9E'
                  for c in centers]
    ax.bar(centers, pcts, color=bar_colors, alpha=0.8, edgecolor='white', width=0.8)
    ax.set_ylabel(label, fontsize=8, fontweight='bold', rotation=0,
                  labelpad=80, va='center', ha='right')
    ax.set_ylim(0, 100)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='-')
    # Mean shift
    m = np.mean(shifts)
    ax.text(3.5, 70, f'avg={m:+.2f}', fontsize=8, color='red',
            fontweight='bold')
    for bar_x, pct in zip(centers, pcts):
        if pct > 3:
            ax.text(bar_x, pct + 1.5, f'{pct:.0f}%', ha='center',
                    fontsize=6, va='bottom')

# Human
plot_shift(axes[0], human_df['IC'].values, human_df['RC'].values, 'Human', '#2196F3')
for spine in axes[0].spines.values():
    spine.set_linewidth(2)
    spine.set_color('#2196F3')

for i, cond in enumerate(SELECTED):
    df = llm_dfs[cond]
    color = colors_map[SIZE_GROUP[cond]]
    plot_shift(axes[i+1], df['IC'].values, df['RC'].values, cond, color)

axes[-1].set_xlabel('RC − IC (negative = revised down, positive = revised up)', fontsize=10)
axes[-1].set_xticks(range(-4, 5))
plt.tight_layout(rect=[0.14, 0, 1, 0.98])
fig.savefig(f'{OUTDIR}/fig8_shift_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'→ Saved fig8_shift_distributions.png')


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Dot plot — 12 shared stories, human mean ± SD vs LLM ratings
# ════════════════════════════════════════════════════════════════════════════

DOT_MODELS = ['Qwen 4B NT', 'Gemma E2B NT', 'Llama 8B',  # small
              'Gemma 31B NT', 'Qwen 27B NT', 'Llama 70B',  # large
              'GPT-5.5', 'Gemini 3.1 Low']  # proprietary
DOT_MODELS = [m for m in DOT_MODELS if m in llm_dfs]

fig, ax = plt.subplots(figsize=(14, 8))

y_positions = np.arange(len(shared_stories))
short_labels = [s.replace('ai_shutdown_', 'AI Shutdown - ')
                 .replace('midnight_store_', 'Store - ')
                 .replace('the_heist_', 'Heist - ')
                 .title()
                for s in shared_stories]

# Human mean ± SD as error bars
h_means = []
h_sds = []
for sid in shared_stories:
    vals = human_df[human_df['story_id'] == sid]['RC'].values
    h_means.append(np.mean(vals))
    h_sds.append(np.std(vals, ddof=1))

ax.errorbar(h_means, y_positions, xerr=h_sds, fmt='s', markersize=10,
            color='#2196F3', capsize=4, capthick=2, linewidth=2,
            label='Human mean ± SD', zorder=10)

# LLM dots
markers = ['o', '^', 'D', 'v', 'P', 'X', '*', 'h']
dot_colors = {
    'Small (≤9B)': ['#66BB6A', '#43A047', '#2E7D32'],
    'Large (≥26B)': ['#FFA726', '#FB8C00', '#EF6C00'],
    'Proprietary': ['#AB47BC', '#8E24AA'],
}
for j, model in enumerate(DOT_MODELS):
    df = llm_dfs[model]
    grp = SIZE_GROUP[model]
    grp_colors = dot_colors[grp]
    color = grp_colors[j % len(grp_colors)]
    llm_rcs = []
    for sid in shared_stories:
        row = df[df['story_id'] == sid]
        llm_rcs.append(row['RC'].values[0] if len(row) == 1 else np.nan)
    # Jitter y slightly
    jitter = (j + 1) * 0.08
    ax.scatter(llm_rcs, y_positions + jitter, marker=markers[j % len(markers)],
               s=50, color=color, alpha=0.8, label=model, zorder=5)

ax.set_yticks(y_positions)
ax.set_yticklabels(short_labels, fontsize=9)
ax.set_xlabel('Reflective Creativity Score (1-7)', fontsize=11)
ax.set_title('12 Shared Stories: Human Ratings vs Selected LLMs\n'
             'Blue squares = human mean ± 1 SD; dots = individual LLM scores',
             fontsize=13, fontweight='bold')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
ax.set_xlim(0.5, 7.5)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUTDIR}/fig9_shared_story_dotplot.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'→ Saved fig9_shared_story_dotplot.png')


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('SUMMARY: DIVERSITY COMPARISON')
print('='*70)

h_rc_stats = scale_stats(human_df['RC'].values)
print(f'\n  {"Group":22s} {"Mean RC SD":>11s} {"Mean Entropy":>13s} '
      f'{"Mean Ceil%":>11s} {"Mean Story SD":>14s}')
print(f'  {"-"*22} {"-"*11} {"-"*13} {"-"*11} {"-"*14}')

print(f'  {"Human":22s} {h_rc_stats["sd"]:11.2f} '
      f'{h_rc_stats["entropy_norm"]:13.3f} '
      f'{h_rc_stats["ceil_pct"]:10.1f}% '
      f'{np.mean(human_sds):14.2f}')

for grp_name, grp_label in [('Small (≤9B)', 'Small LLMs (≤9B)'),
                              ('Large (≥26B)', 'Large LLMs (≥26B)'),
                              ('Proprietary', 'Proprietary')]:
    grp_models = [k for k in llm_dfs if SIZE_GROUP.get(k) == grp_name]
    sds = [scale_stats(llm_dfs[m]['RC'].values)['sd'] for m in grp_models]
    ents = [scale_stats(llm_dfs[m]['RC'].values)['entropy_norm'] for m in grp_models]
    ceils = [scale_stats(llm_dfs[m]['RC'].values)['ceil_pct'] for m in grp_models]

    # Story SD for this group
    story_sds_grp = []
    for sid in shared_stories:
        scores = []
        for m in grp_models:
            row = llm_dfs[m][llm_dfs[m]['story_id'] == sid]
            if len(row) == 1:
                scores.append(row['RC'].values[0])
        if len(scores) > 1:
            story_sds_grp.append(np.std(scores, ddof=1))

    print(f'  {grp_label:22s} {np.mean(sds):11.2f} '
          f'{np.mean(ents):13.3f} '
          f'{np.mean(ceils):10.1f}% '
          f'{np.mean(story_sds_grp):14.2f}')

print('\nInterpretation:')
print('  Higher entropy = uses more of the scale = more diverse ratings')
print('  Higher story SD = models disagree more on which stories are creative')
print('  Human-like = high entropy + high story SD + moderate ceiling%')
