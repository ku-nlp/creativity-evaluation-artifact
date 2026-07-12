"""Compute IC β-vector pathway similarity for each LLM condition,
alongside RC for comparison."""
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'study2_llm/data/raw_json'
HUMAN_CSV = ROOT / 'study1_human/data/human_ratings.csv'

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


def cosine(a, b):
    a, b = np.array(a), np.array(b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den else 0.0


def betas(X, y):
    """Standardised β: regress y (centred) on z-scored X."""
    X_z = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
    m = LinearRegression().fit(X_z, y)
    return m.coef_


# Human β
df = pd.read_csv(HUMAN_CSV)
df = df.rename(columns={'O_Initial_Creativity': 'IC', 'O_Final_Creativity': 'RC'})
X_h = df[HUMAN_KEYS].values.astype(float)
human_beta_ic = betas(X_h, df['IC'].values.astype(float))
human_beta_rc = betas(X_h, df['RC'].values.astype(float))

print(f"{'Condition':18s} {'cos(IC)':>8s} {'cos(RC)':>8s}")
print('-' * 40)
results = []
for label, fname in LLM_FILES.items():
    path = RAW / fname
    if not path.exists():
        print(f'MISSING: {label}')
        continue
    with open(path) as f:
        data = json.load(f)
    rows = []
    for r in data['results']:
        if not r.get('parse_ok', True): continue
        row = {'IC': r['scores']['initial_creativity'],
               'RC': r['scores']['reflective_creativity']}
        for k in SUB_KEYS:
            row[k] = r['scores']['sub_components'][k]
        rows.append(row)
    dfl = pd.DataFrame(rows)
    X_l = dfl[SUB_KEYS].values.astype(float)
    if X_l.std() == 0:
        continue
    beta_ic = betas(X_l, dfl['IC'].values.astype(float))
    beta_rc = betas(X_l, dfl['RC'].values.astype(float))
    cos_ic = cosine(human_beta_ic, beta_ic)
    cos_rc = cosine(human_beta_rc, beta_rc)
    results.append((label, cos_ic, cos_rc))
    print(f'{label:18s} {cos_ic:>+8.3f} {cos_rc:>+8.3f}')

# Sort by cos(RC) descending for final output
results.sort(key=lambda r: -r[2])
print('\nSorted by cos(RC):')
for i, (label, ci, cr) in enumerate(results, 1):
    print(f'{i:>2d}. {label:18s} cos(IC)={ci:+.3f}  cos(RC)={cr:+.3f}')
