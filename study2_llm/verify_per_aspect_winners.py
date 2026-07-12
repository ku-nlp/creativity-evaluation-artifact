"""Verify per-aspect Spearman winners (Table 8 / tab:winners in paper).

For each of the 11 sub-components + RC + IC, compute Spearman ρ between
the human mean-per-story ranking and the LLM ranking on the 12 shared stories.
Report the winning LLM condition per dimension.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
HUMAN_CSV = ROOT / 'study1_human/data/human_ratings.csv'
RAW = ROOT / 'study2_llm/data/raw_json'

TOPIC_TO_PREFIX = {
    'An advanced AI initiates its own permanent shutdown sequence': 'ai_shutdown',
    'A lone worker in a Japanese convenience store at 3:00 AM. The city is silent.': 'midnight_store',
    'A professional thief attempting to crack a high-security safe in a dark room. High tension.': 'the_heist',
}

HUMAN_SUB_MAP = {
    'R_Emotion': 'emotional_impact', 'A_Topic': 'topic_fidelity',
    'N_Vocab': 'vocabulary_freshness', 'N_Plot': 'plot_uniqueness',
    'N_Surprise': 'surprise', 'R_Empathy': 'empathy',
    'R_Thought': 'thought_provocation', 'V_Engagement': 'engagement',
    'V_Style': 'stylistic_quality', 'V_Logic': 'logical_coherence',
    'A_Tone': 'tone_fidelity',
}

SUB_KEYS = list(HUMAN_SUB_MAP.values())
HUMAN_KEYS = list(HUMAN_SUB_MAP.keys())

LLM_FILES = {
    'Qwen3.5-4B-NT':    'results_qwen35_4b_CLEAN_nothink.json',
    'Qwen3.5-4B-T':     'results_qwen35_4b_CLEAN_think.json',
    'Qwen3.5-9B-NT':    'results_qwen35_9b_CLEAN_nothink.json',
    'Qwen3.5-9B-T':     'results_qwen35_9b_CLEAN_think.json',
    'Qwen3.5-27B-NT':   'results_qwen35_27b_CLEAN_nothink.json',
    'Qwen3.5-27B-T':    'results_qwen35_27b_CLEAN_think.json',
    'Qwen3.5-122B-NT':  'results_qwen35_122b_judge_gemini_nothink.json',
    'Qwen3.5-122B-T':   'results_qwen35_122b_judge_gemini_think.json',
    'Llama3.1-8B':      'results_llama31_8b_judge_gemini_nothink.json',
    'Llama3.1-70B':     'results_llama31_70b_judge_gemini_nothink.json',
    'Gemma4-E2B-NT':    'results_gemma4_e2b_nothink.json',
    'Gemma4-E2B-T':     'results_gemma4_e2b_think.json',
    'Gemma4-E4B-NT':    'results_gemma4_e4b_nothink.json',
    'Gemma4-E4B-T':     'results_gemma4_e4b_think.json',
    'Gemma4-26B-NT':    'results_gemma4_26b_nothink.json',
    'Gemma4-26B-T':     'results_gemma4_26b_think.json',
    'Gemma4-31B-NT':    'results_gemma4_31b_nothink.json',
    'Gemma4-31B-T':     'results_gemma4_31b_think.json',
    'Gemini3.1-Low':    'results_gemini_31_pro_low.json',
    'Gemini3.1-Med':    'results_gemini_31_pro_medium.json',
    'Gemini3.1-High':   'results_gemini_31_pro_high.json',
    'Gemini3-Low':      'results_gemini_3_pro_low.json',
    'Gemini3-Med':      'results_gemini_3_pro_medium.json',
    'Gemini3-High':     'results_gemini_3_pro_high.json',
    'GPT-5.5':          'results_gpt55_CLEAN_high.json',
}


def load_human():
    df = pd.read_csv(HUMAN_CSV)
    df = df.rename(columns={'O_Initial_Creativity': 'IC', 'O_Final_Creativity': 'RC'})
    df['story_id'] = df.apply(lambda r: TOPIC_TO_PREFIX[r['TOPIC']] + '_' + r['TONE'].lower(), axis=1)
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


human = load_human()
shared = sorted(human['story_id'].unique())
assert len(shared) == 12, f'Expected 12 shared stories, got {len(shared)}'

# Build per-story human means for IC, RC, and each sub-component
human_means = {}
for col in ['IC', 'RC'] + HUMAN_KEYS:
    means = human.groupby('story_id')[col].mean().reindex(shared)
    human_means[col] = means

# For each LLM condition, compute Spearman ρ on each dimension
DIMENSIONS = ['RC', 'IC'] + HUMAN_KEYS  # 13 dimensions

results = {dim: [] for dim in DIMENSIONS}

for label, fname in LLM_FILES.items():
    path = RAW / fname
    if not path.exists():
        print(f'MISSING: {label} ({path})')
        continue
    df = load_llm(path)
    df = df[df['story_id'].isin(shared)]
    if len(df) < 12:
        print(f'SKIP {label}: only {len(df)} of 12 shared stories')
        continue
    df = df.set_index('story_id').loc[shared]
    for dim in DIMENSIONS:
        if dim in HUMAN_KEYS:
            llm_col = HUMAN_SUB_MAP[dim]
        else:
            llm_col = dim
        llm_vals = df[llm_col].values
        h_vals = human_means[dim].values
        # Check for constant LLM vector (zero variance)
        if np.std(llm_vals) == 0:
            rho = np.nan
        else:
            rho, _ = spearmanr(h_vals, llm_vals)
        results[dim].append((label, rho))

# Winners per dimension
print('\n' + '='*70)
print('PER-DIMENSION WINNERS (Spearman ρ vs human, n=12 shared stories)')
print(f"Significance threshold |ρ| > 0.58 at p<0.05")
print('='*70)
print(f'{"Dimension":15s} | {"Winner":18s} | {"ρ":>7s} | All conditions (top 5)')
print('-'*100)
for dim in DIMENSIONS:
    sorted_results = sorted(results[dim], key=lambda x: -x[1] if not np.isnan(x[1]) else -999)
    sorted_results = [(l, r) for l, r in sorted_results if not np.isnan(r)]
    if not sorted_results:
        continue
    winner_label, winner_rho = sorted_results[0]
    top5 = ', '.join(f'{l}:{r:+.2f}' for l, r in sorted_results[:5])
    print(f'{dim:15s} | {winner_label:18s} | {winner_rho:+7.3f} | {top5}')
