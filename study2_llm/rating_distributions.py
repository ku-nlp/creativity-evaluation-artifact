"""
Rating distribution plots: Human (115 raters) vs each LLM family (all 80 stories).
Shows IC and RC as side-by-side histograms on the 1–7 scale.
Output: study2_llm/analysis_figures/fig_rating_distributions.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
OUT_DIR = Path('study2_llm/analysis_figures')
OUT_DIR.mkdir(exist_ok=True)

# ── Condition map (same as cross_analysis.py) ───────────────────────────────
LLM_CONDITIONS = {
    'Qwen3.5-4B-NT':    'study2_llm/data/raw_json/results_qwen35_4b_CLEAN_nothink.json',
    'Qwen3.5-4B-T':     'study2_llm/data/raw_json/results_qwen35_4b_CLEAN_think.json',
    'Qwen3.5-9B-NT':    'study2_llm/data/raw_json/results_qwen35_9b_CLEAN_nothink.json',
    'Qwen3.5-9B-T':     'study2_llm/data/raw_json/results_qwen35_9b_CLEAN_think.json',
    'Qwen3.5-27B-NT':   'study2_llm/data/raw_json/results_qwen35_27b_CLEAN_nothink.json',
    'Qwen3.5-27B-T':    'study2_llm/data/raw_json/results_qwen35_27b_CLEAN_think.json',
    'Qwen3.5-122B-NT':  'study2_llm/data/raw_json/results_qwen35_122b_judge_gemini_nothink.json',
    'Qwen3.5-122B-T':   'study2_llm/data/raw_json/results_qwen35_122b_judge_gemini_think.json',
    'Llama3.1-8B':      'study2_llm/data/raw_json/results_llama31_8b_judge_gemini_nothink.json',
    'Llama3.1-70B':     'study2_llm/data/raw_json/results_llama31_70b_judge_gemini_nothink.json',
    'Gemma4-E2B-NT':    'study2_llm/data/raw_json/results_gemma4_e2b_nothink.json',
    'Gemma4-E2B-T':     'study2_llm/data/raw_json/results_gemma4_e2b_think.json',
    'Gemma4-E4B-NT':    'study2_llm/data/raw_json/results_gemma4_e4b_nothink.json',
    'Gemma4-E4B-T':     'study2_llm/data/raw_json/results_gemma4_e4b_think.json',
    'Gemma4-26B-NT':    'study2_llm/data/raw_json/results_gemma4_26b_nothink.json',
    'Gemma4-26B-T':     'study2_llm/data/raw_json/results_gemma4_26b_think.json',
    'Gemma4-31B-NT':    'study2_llm/data/raw_json/results_gemma4_31b_nothink.json',
    'Gemma4-31B-T':     'study2_llm/data/raw_json/results_gemma4_31b_think.json',
    'Gemini3.1-Low':    'study2_llm/data/raw_json/results_gemini_31_pro_low.json',
    'Gemini3.1-Med':    'study2_llm/data/raw_json/results_gemini_31_pro_medium.json',
    'Gemini3.1-High':   'study2_llm/data/raw_json/results_gemini_31_pro_high.json',
    'Gemini3-Low':      'study2_llm/data/raw_json/results_gemini_3_pro_low.json',
    'Gemini3-Med':      'study2_llm/data/raw_json/results_gemini_3_pro_medium.json',
    'Gemini3-High':     'study2_llm/data/raw_json/results_gemini_3_pro_high.json',
    'GPT-5.5':          'study2_llm/data/raw_json/results_gpt55_CLEAN_high.json',
}

# Family groupings (condition label prefix → family name)
FAMILIES = {
    'Qwen':   [k for k in LLM_CONDITIONS if k.startswith('Qwen')],
    'Llama':  [k for k in LLM_CONDITIONS if k.startswith('Llama')],
    'Gemma':  [k for k in LLM_CONDITIONS if k.startswith('Gemma')],
    'Gemini': [k for k in LLM_CONDITIONS if k.startswith('Gemini')],
    'GPT-5.5': ['GPT-5.5'],
}

# ── Loaders ─────────────────────────────────────────────────────────────────
def load_human():
    df = pd.read_csv('study1_human/data/human_ratings.csv')
    ic = df['O_Initial_Creativity'].dropna().astype(int).tolist()
    rc = df['O_Final_Creativity'].dropna().astype(int).tolist()
    return ic, rc

def load_llm_condition(path):
    with open(path) as f:
        d = json.load(f)
    results = d if isinstance(d, list) else d.get('results', [])
    ic, rc = [], []
    for r in results:
        if not r.get('parse_ok', True):
            continue
        s = r.get('scores', {})
        if s.get('initial_creativity') and s.get('reflective_creativity'):
            ic.append(int(s['initial_creativity']))
            rc.append(int(s['reflective_creativity']))
    return ic, rc

def load_family(conditions):
    all_ic, all_rc = [], []
    for cond in conditions:
        path = LLM_CONDITIONS[cond]
        ic, rc = load_llm_condition(path)
        all_ic.extend(ic)
        all_rc.extend(rc)
    return all_ic, all_rc

# ── Load all data ─────────────────────────────────────────────────────────
print("Loading data...")
human_ic, human_rc = load_human()
family_data = {}
for fam, conds in FAMILIES.items():
    fam_ic, fam_rc = load_family(conds)
    family_data[fam] = (fam_ic, fam_rc)
    n_stories = len(fam_ic)
    print(f"  {fam}: {n_stories} IC scores, {len(conds)} condition(s)")

print(f"  Human: {len(human_ic)} IC scores (115 raters)")

# ── Color palette ─────────────────────────────────────────────────────────
IC_COLOR = '#4C72B0'   # blue
RC_COLOR = '#DD8452'   # orange
SCALE = range(1, 8)    # 1–7

def score_pct(scores):
    """Return % for each score 1–7."""
    n = len(scores)
    return [100 * scores.count(v) / n for v in SCALE]

# ── Build figure ─────────────────────────────────────────────────────────
rows = [('Human\n(115 raters)', human_ic, human_rc)] + \
       [(fam, *family_data[fam]) for fam in FAMILIES]

n_rows = len(rows)
fig, axes = plt.subplots(n_rows, 2, figsize=(11, 2.6 * n_rows),
                          constrained_layout=True)

BAR_W = 0.55
X = np.array(list(SCALE))

# Column headers on top row only
axes[0, 0].set_title('Initial Creativity (IC)', fontsize=10, fontweight='bold', pad=6)
axes[0, 1].set_title('Reflective Creativity (RC)', fontsize=10, fontweight='bold', pad=6)

for row_idx, (label, ic, rc) in enumerate(rows):
    ax_ic = axes[row_idx, 0]
    ax_rc = axes[row_idx, 1]

    pct_ic = score_pct(ic)
    pct_rc = score_pct(rc)
    ic_mean = np.mean(ic)
    rc_mean = np.mean(rc)
    ic_sd   = np.std(ic)
    rc_sd   = np.std(rc)

    # IC panel
    ax_ic.bar(X, pct_ic, BAR_W, color=IC_COLOR, alpha=0.85)
    ax_ic.axvline(ic_mean, color='#1a3a6b', linewidth=2.0, linestyle='--', alpha=0.9)
    # Stats annotation inside panel (top-left)
    ax_ic.text(0.03, 0.93, f'μ={ic_mean:.2f}  SD={ic_sd:.2f}',
               transform=ax_ic.transAxes, fontsize=7.5, va='top', color='#1a3a6b')

    # Row label (left of IC panel only)
    ax_ic.set_ylabel(label, fontsize=9, rotation=0, labelpad=72,
                     va='center', ha='right', fontweight='bold')

    # RC panel
    ax_rc.bar(X, pct_rc, BAR_W, color=RC_COLOR, alpha=0.85)
    ax_rc.axvline(rc_mean, color='#7a3a10', linewidth=2.0, linestyle='--', alpha=0.9)
    ax_rc.text(0.03, 0.93, f'μ={rc_mean:.2f}  SD={rc_sd:.2f}',
               transform=ax_rc.transAxes, fontsize=7.5, va='top', color='#7a3a10')

    for ax in [ax_ic, ax_rc]:
        ax.set_xlim(0.5, 7.5)
        ax.set_xticks(list(SCALE))
        ax.set_xticklabels([str(v) for v in SCALE], fontsize=8)
        ax.set_ylim(0, 80)
        ax.set_yticks([0, 25, 50, 75])
        ax.yaxis.set_tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linewidth=0.5, alpha=0.35)

    # n label (top-right, small)
    ax_rc.text(0.98, 0.93, f'n={len(ic)}', transform=ax_rc.transAxes,
               fontsize=7, ha='right', va='top', color='#666')

# x-axis label on bottom row only
axes[-1, 0].set_xlabel('Score (1–7)', fontsize=9)
axes[-1, 1].set_xlabel('Score (1–7)', fontsize=9)

# Global y-axis label
fig.text(0.005, 0.5, '% of ratings', va='center', rotation='vertical', fontsize=10)

fig.suptitle('Rating Distributions: Human vs LLM Families',
             fontsize=12, fontweight='bold', y=1.005)

out_path = OUT_DIR / 'fig_rating_distributions.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved → {out_path}")

# ── Print summary table ───────────────────────────────────────────────────
print("\nSummary:")
print(f"{'Source':<15} {'IC mean':>8} {'IC SD':>7} {'RC mean':>8} {'RC SD':>7} "
      f"{'% score 7':>10} {'n':>6}")
print("-" * 65)

def stats(scores):
    arr = np.array(scores)
    return arr.mean(), arr.std(), 100 * (arr == 7).sum() / len(arr)

h_ic_m, h_ic_s, h_ic_ceil = stats(human_ic)
h_rc_m, h_rc_s, h_rc_ceil = stats(human_rc)
print(f"{'Human':<15} {h_ic_m:>8.2f} {h_ic_s:>7.2f} {h_rc_m:>8.2f} {h_rc_s:>7.2f} "
      f"{h_rc_ceil:>9.1f}% {len(human_rc):>6}")

for fam, (fam_ic, fam_rc) in family_data.items():
    ic_m, ic_s, ic_ceil = stats(fam_ic)
    rc_m, rc_s, rc_ceil = stats(fam_rc)
    n_conds = len(FAMILIES[fam])
    print(f"{fam:<15} {ic_m:>8.2f} {ic_s:>7.2f} {rc_m:>8.2f} {rc_s:>7.2f} "
          f"{rc_ceil:>9.1f}% {len(fam_rc):>6}  ({n_conds} cond)")
