import json
import os
from pathlib import Path

def check_integrity(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    model = data.get('model', 'unknown')
    mode = "Think" if data.get('thinking_mode') else "Nothink"
    
    unique_stories = set()
    corrupt_entries = []
    failed_parses = 0
    missing_scores = 0
    invalid_ranges = 0
    empty_reasoning = 0
    
    expected_subkeys = [
        "emotional_impact", "topic_fidelity", "vocabulary_freshness", "plot_uniqueness",
        "surprise", "empathy", "thought_provocation", "engagement",
        "stylistic_quality", "logical_coherence", "tone_fidelity"
    ]

    for r in results:
        story_id = r.get('story_id')
        unique_stories.add(story_id)
        
        if not r.get('parse_ok'):
            failed_parses += 1
            continue
            
        scores = r.get('scores', {})
        
        # Check Turn 1/3 scores
        ic = scores.get('initial_creativity')
        rc = scores.get('reflective_creativity')
        if ic is None or rc is None:
            missing_scores += 1
            corrupt_entries.append((story_id, "Missing IC/RC"))
            continue
            
        if not (1 <= ic <= 10) or not (1 <= rc <= 10): # Allowing up to 10 just in case of slight over-range, though 7 is standard
            invalid_ranges += 1
            corrupt_entries.append((story_id, f"Invalid IC/RC range: {ic}/{rc}"))

        # Check Sub-components
        sub = scores.get('sub_components', {})
        if not sub:
            missing_scores += 1
            corrupt_entries.append((story_id, "Missing sub_components"))
        else:
            for k in expected_subkeys:
                if k not in sub:
                    missing_scores += 1
                    corrupt_entries.append((story_id, f"Missing subkey: {k}"))
                    break
        
        # Check Reasoning
        raw_r3 = r.get('raw_responses', {}).get('turn3', '')
        if '"reasoning":' not in raw_r3 and '"reasoning": ""' in raw_r3:
            empty_reasoning += 1

    return {
        "model": model,
        "mode": mode,
        "total": len(results),
        "unique": len(unique_stories),
        "failed_parses": failed_parses,
        "missing_scores": missing_scores,
        "invalid_ranges": invalid_ranges,
        "corrupt_list": corrupt_entries[:5] # Show first few
    }

def main():
    raw_json_dir = Path("study2_llm/data/raw_json/")
    files = list(raw_json_dir.glob("*.json"))
    
    print(f"{'File':<50} | {'Uniq'} | {'Fail'} | {'Miss'} | {'Inv'} | {'Status'}")
    print("-" * 90)
    
    for f in sorted(files):
        try:
            stats = check_integrity(f)
            status = "OK" if stats['unique'] >= 80 and stats['failed_parses'] == 0 and stats['missing_scores'] == 0 else "ISSUE"
            print(f"{f.name:<50} | {stats['unique']:<4} | {stats['failed_parses']:<4} | {stats['missing_scores']:<4} | {stats['invalid_ranges']:<4} | {status}")
            if stats['corrupt_list']:
                for sid, msg in stats['corrupt_list']:
                    print(f"   -> {sid}: {msg}")
        except Exception as e:
            print(f"{f.name:<50} | ERROR: {e}")

if __name__ == "__main__":
    main()
