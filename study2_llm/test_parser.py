import json
import re
from pathlib import Path
from typing import Optional, Dict, List

# Mocking the functions from run_judges.py for testing
def strip_think_block(text: str) -> str:
    """Remove <think>...</think> blocks and common 'Thinking Process' headers."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    # Also remove "Thinking Process" or "Thought Process" style headers
    text = re.sub(r"^(?:Thinking|Thought)\s+Process:?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()

def parse_json(content: str) -> Optional[dict]:
    content = strip_think_block(content)
    content = re.sub(r"```(?:json)?|```", "", content)

    starts = [m.start() for m in re.finditer(r'\{', content)]
    for s in reversed(starts):
        ends = [m.start() for m in re.finditer(r'\}', content[s:])]
        for e in reversed(ends):
            candidate = content[s : s + e + 1].strip()
            try:
                return json.loads(candidate)
            except:
                continue
    return None

def _fuzzy_match_key(key: str, canonical: str) -> bool:
    """Check if a mangled key matches the canonical one using edit distance."""
    k = key.replace(" ", "").replace("_", "").lower()
    c = canonical.replace("_", "").lower()
    if k == c:
        return True
    # First-word match (e.g. "tonal_fidelity" -> "tone_fidelity": both start with "ton")
    k_first = key.replace(" ", "").split("_")[0].lower()
    c_first = canonical.split("_")[0].lower()
    # Share a 3-char prefix on the first word + similar length overall
    if (len(k_first) >= 3 and len(c_first) >= 3
            and k_first[:3] == c_first[:3]
            and abs(len(k) - len(c)) <= 5):
        return True
    return False

def fix_keys(data: dict, canonical_keys: list) -> dict:
    """Map mangled keys to canonical ones using fuzzy matching."""
    if data is None:
        return data
    fixed = {}
    used = set()
    for ck in canonical_keys:
        if ck in data:
            fixed[ck] = data[ck]
            used.add(ck)
        else:
            for key in data:
                if key not in used and _fuzzy_match_key(key, ck):
                    fixed[ck] = data[key]
                    used.add(key)
                    break
    return fixed

TURN1_KEYS = ["initial_creativity", "enjoyment"]
TURN2_KEYS = ["emotional_impact", "topic_fidelity", "vocabulary_freshness", "plot_uniqueness",
              "surprise", "empathy", "thought_provocation", "engagement",
              "stylistic_quality", "logical_coherence", "tone_fidelity"]

def validate_turn1(data: dict) -> bool:
    return data is not None and set(TURN1_KEYS).issubset(data.keys())

def validate_turn2(data: dict) -> bool:
    return data is not None and set(TURN2_KEYS).issubset(data.keys())

def validate_turn3(data: dict) -> bool:
    return data is not None and "reflective_creativity" in data

def main():
    results_path = Path("study2_llm/data/raw_json/results_qwen122b_judge_gemini.json")
    if not results_path.exists():
        print(f"File not found: {results_path}")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    failed = [r for r in results if not r.get("parse_ok")]
    
    print(f"Total results: {len(results)}")
    print(f"Failed results: {len(failed)}")
    
    rescued = 0
    for r in failed:
        raws = r.get("raw_responses", {})
        story_id = r.get("story_id")
        
        ok = True
        parsed_all = {}
        print(f"\nAttempting rescue: {story_id}")
        for turn in ["turn1", "turn2", "turn3"]:
            raw = raws.get(turn, "")
            parsed = parse_json(raw)
            
            if turn == "turn1":
                parsed = fix_keys(parsed, TURN1_KEYS)
                if not validate_turn1(parsed):
                    print(f"  {turn} rescue failed. Parsed: {parsed}")
                    if parsed is None or parsed == {}:
                        print(f"    Raw sample: {raw[:200]}...")
                    ok = False
                    # break # Check all turns for debug
            elif turn == "turn2":
                parsed = fix_keys(parsed, TURN2_KEYS)
                if not validate_turn2(parsed):
                    print(f"  {turn} rescue failed. Parsed: {parsed}")
                    if parsed is None or parsed == {}:
                        print(f"    Raw sample: {raw[:200]}...")
                    ok = False
                    # break
            elif turn == "turn3":
                if not validate_turn3(parsed):
                    print(f"  {turn} rescue failed. Parsed: {parsed}")
                    if parsed is None or parsed == {}:
                        print(f"    Raw sample: {raw[:200]}...")
                    ok = False
                    # break
            
            parsed_all[turn] = parsed
            
        if ok:
            rescued += 1
            print(f"  SUCCESSFULLY RESCUED: {story_id}")
            
    print(f"\nSummary: Rescued {rescued} out of {len(failed)} failures.")

if __name__ == "__main__":
    main()
