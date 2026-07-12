"""
Prompt Sensitivity Test — Turn 3 Variants
Runs all 4 Turn 3 variants on all 12 stories using one model.
Saves results to study2_llm/prompt_sensitivity/results/

Usage:
    python experiments_v2/prompt_sensitivity/run_sensitivity.py
"""

import sys, os, json, re, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openai import OpenAI
from study2_llm.rating_scripts.shared_prompts import STORIES, SYSTEM_PROMPT, build_turn1_prompt, build_turn2_prompt, build_turn3_variant

# ── Config ────────────────────────────────────────────────────────────────────
MODEL    = "gpt-4.1-mini"   # fast + cheap for sensitivity test
VARIANTS = ["v1", "v2", "v3", "v4"]
OUT_DIR  = "study2_llm/prompt_sensitivity/results"
os.makedirs(OUT_DIR, exist_ok=True)

client = OpenAI()  # reads OPENAI_API_KEY from env

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_json(text: str) -> dict:
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)

def chat(messages: list) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()

def run_story_variant(story: dict, variant: str) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Turn 1
    t1 = build_turn1_prompt(story)
    messages.append({"role": "user", "content": t1})
    r1_raw = chat(messages)
    messages.append({"role": "assistant", "content": r1_raw})
    r1 = parse_json(r1_raw)

    # Turn 2
    t2 = build_turn2_prompt()
    messages.append({"role": "user", "content": t2})
    r2_raw = chat(messages)
    messages.append({"role": "assistant", "content": r2_raw})
    r2 = parse_json(r2_raw)

    # Turn 3 — variant
    t3 = build_turn3_variant(variant)
    messages.append({"role": "user", "content": t3})
    r3_raw = chat(messages)
    r3 = parse_json(r3_raw)

    return {
        "story_id": story["id"],
        "topic":    story["topic"],
        "tone":     story["tone"],
        "variant":  variant,
        "IC":       r1.get("initial_creativity"),
        "enjoyment":r1.get("enjoyment"),
        "RC":       r3.get("reflective_creativity"),
        "reasoning":r3.get("reasoning", ""),
        "flipped":  r1.get("initial_creativity") != r3.get("reflective_creativity"),
        "IC_RC_diff": (r3.get("reflective_creativity", 0) - r1.get("initial_creativity", 0)),
        "sub": r2,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
results = []

for variant in VARIANTS:
    print(f"\n── Variant {variant.upper()} ──────────────────────")
    for story in STORIES:
        try:
            result = run_story_variant(story, variant)
            results.append(result)
            flip_marker = "↕" if result["flipped"] else "="
            print(f"  {story['id']:35s} IC={result['IC']} RC={result['RC']} {flip_marker}  | {result['reasoning'][:80]}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  ERROR {story['id']} variant={variant}: {e}")

# Save raw results
out_path = os.path.join(OUT_DIR, f"sensitivity_{MODEL.replace('.','_')}.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} results → {out_path}")
