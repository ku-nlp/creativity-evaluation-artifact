"""
Prompt Sensitivity Test — Gemini 2.5 Pro, all 4 Turn 3 variants
Runs 4 variants × 12 stories = 48 conversations

Usage:
    python experiments_v2/prompt_sensitivity/run_sensitivity_gemini.py
"""

import sys, os, json, re, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from google import genai
from google.genai import types
from study2_llm.rating_scripts.shared_prompts import STORIES, SYSTEM_PROMPT, build_turn1_prompt, build_turn2_prompt, build_turn3_variant

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID  = "gemini-2.5-pro"
VARIANTS  = ["v1", "v2", "v3", "v4"]
OUT_DIR   = "study2_llm/prompt_sensitivity/results"
os.makedirs(OUT_DIR, exist_ok=True)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_json(text: str):
    if not text:
        return None
    text = re.sub(r"```json|```", "", text).strip()
    # Try full parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting first {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None

def extract_text(response) -> str:
    try:
        text = response.text
        if text:
            return text
    except Exception:
        pass
    if response.candidates:
        parts = response.candidates[0].content.parts
        return "".join(p.text for p in parts if hasattr(p, "text") and p.text)
    return ""

def run_story_variant(story: dict, variant: str) -> dict:
    gen_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=16000,
    )
    chat = client.chats.create(model=MODEL_ID, config=gen_config)

    # Turn 1
    r1_raw = extract_text(chat.send_message(build_turn1_prompt(story)))
    r1 = parse_json(r1_raw) or {}

    # Turn 2
    r2_raw = extract_text(chat.send_message(build_turn2_prompt()))
    r2 = parse_json(r2_raw) or {}

    # Turn 3 — variant
    r3_raw = extract_text(chat.send_message(build_turn3_variant(variant)))
    r3 = parse_json(r3_raw) or {}

    ic = r1.get("initial_creativity")
    rc = r3.get("reflective_creativity")

    return {
        "story_id":  story["id"],
        "tone":      story["tone"],
        "variant":   variant,
        "IC":        ic,
        "RC":        rc,
        "reasoning": r3.get("reasoning", ""),
        "flipped":   ic != rc,
        "IC_RC_diff": (rc - ic) if (rc is not None and ic is not None) else None,
        "sub":       r2,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
results = []

for variant in VARIANTS:
    print(f"\n── Variant {variant.upper()} ({MODEL_ID}) ──────────────────────")
    for story in STORIES:
        try:
            result = run_story_variant(story, variant)
            results.append(result)
            flip_marker = "↕" if result["flipped"] else "="
            reasoning_preview = result["reasoning"][:80] if result["reasoning"] else "-"
            print(f"  {story['id']:35s} IC={result['IC']} RC={result['RC']} {flip_marker}  | {reasoning_preview}")
            time.sleep(1.5)  # avoid rate limits
        except Exception as e:
            print(f"  ERROR {story['id']} variant={variant}: {e}")
            time.sleep(3)

# Save
out_path = os.path.join(OUT_DIR, f"sensitivity_{MODEL_ID.replace('-','_')}.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} results → {out_path}")
