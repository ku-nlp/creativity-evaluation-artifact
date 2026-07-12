"""Per-condition × per-story-type revision direction (mean shift cond on revising)."""
import json, numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'study2_llm/data/raw_json'

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

TYPE_MAP = {
    'surreal': 'high-novelty',
    'clinical': 'high-adherence',
    'witty': 'high-value',
    'melancholic': 'high-resonance',
}
TYPES_ORDER = ['high-adherence', 'high-novelty', 'high-value', 'high-resonance']


def story_type(sid):
    # story_id like 'ai_shutdown_clinical' — last token is the original tone label
    suffix = sid.rsplit('_', 1)[-1].lower()
    return TYPE_MAP.get(suffix, None)


def per_type_stats(path):
    with open(path) as f:
        data = json.load(f)
    by_type = defaultdict(list)
    n_total = 0
    for r in data['results']:
        if not r.get('parse_ok', True):
            continue
        sid = r['story_id']
        st = story_type(sid)
        if st is None:
            continue
        ic = r['scores']['initial_creativity']
        rc = r['scores']['reflective_creativity']
        delta = rc - ic
        by_type[st].append(delta)
        n_total += 1
    out = {}
    revised_total = 0
    for st in TYPES_ORDER:
        deltas = by_type[st]
        revised = [d for d in deltas if d != 0]
        n = len(deltas)
        rev_count = len(revised)
        revised_total += rev_count
        mean_shift = np.mean(revised) if revised else np.nan
        out[st] = {'n': n, 'rev_count': rev_count, 'mean_shift': mean_shift}
    overall_rev_pct = 100 * revised_total / n_total if n_total else 0
    return overall_rev_pct, out


print(f'{"Condition":18s} | {"Rev%":>5s} | {"hi-A":>6s} {"hi-N":>6s} {"hi-V":>6s} {"hi-R":>6s}')
print('-' * 70)
for label, fname in LLM_FILES.items():
    path = RAW / fname
    if not path.exists():
        print(f'MISSING: {label}')
        continue
    rev_pct, stats = per_type_stats(path)
    cells = []
    for st in TYPES_ORDER:
        ms = stats[st]['mean_shift']
        if np.isnan(ms):
            cells.append('   -- ')
        else:
            cells.append(f'{ms:+5.2f}')
    print(f'{label:18s} | {rev_pct:>4.1f}% | {cells[0]:>6s} {cells[1]:>6s} {cells[2]:>6s} {cells[3]:>6s}')
