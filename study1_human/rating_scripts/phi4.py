"""
conv_phi4.py — Conversational 3-turn experiment: Phi-4 (standard) + Phi-4-reasoning

Runs against whichever model is loaded in vLLM. Auto-detects model ID.
Run once with phi-4 loaded, once with Phi-4-reasoning loaded.

12 stories × 3 turns = 36 API calls per run.

Run:
    python conv_phi4.py

vLLM (standard):
    python -m vllm.entrypoints.openai.api_server --model microsoft/phi-4

vLLM (reasoning):
    python -m vllm.entrypoints.openai.api_server --model microsoft/Phi-4-reasoning
"""

import json
import time
import re
import requests
from datetime import datetime
from pathlib import Path
from openai import OpenAI

from shared_stories import (
    STORIES, SYSTEM_PROMPT,
    build_turn1_prompt, build_turn2_prompt, build_turn3_prompt,
)

VLLM_BASE_URL = "http://localhost:8000/v1"
TEMPERATURE = 0.0
MAX_TOKENS = 1024
MAX_TOKENS_REASONING = 16384  # reasoning model needs room for thinking + response
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "conv_results"

# Model ID -> output file mapping
MODEL_CONFIG = {
    "microsoft/phi-4": {
        "label": "Phi-4 (standard)",
        "output_file": "phi4_standard.json",
        "max_tokens": MAX_TOKENS,
        "is_reasoning": False,
    },
    "microsoft/Phi-4-reasoning": {
        "label": "Phi-4 (reasoning)",
        "output_file": "phi4_reasoning.json",
        "max_tokens": MAX_TOKENS_REASONING,
        "is_reasoning": True,
    },
    "microsoft/phi-4-reasoning": {
        "label": "Phi-4 (reasoning)",
        "output_file": "phi4_reasoning.json",
        "max_tokens": MAX_TOKENS_REASONING,
        "is_reasoning": True,
    },
}

def detect_model():
    """Auto-detect which model is loaded in vLLM."""
    resp = requests.get(f"{VLLM_BASE_URL}/models")
    models = resp.json()["data"]
    model_id = models[0]["id"]

    if model_id in MODEL_CONFIG:
        return model_id, MODEL_CONFIG[model_id]

    # Fuzzy match
    for known_id, config in MODEL_CONFIG.items():
        if known_id.lower() in model_id.lower() or model_id.lower() in known_id.lower():
            print(f"Matched vLLM model '{model_id}' to config '{known_id}'")
            config = dict(config)  # copy
            return model_id, config

    # Unknown model — use defaults
    print(f"WARNING: Unknown model '{model_id}'. Using standard config.")
    return model_id, {
        "label": model_id,
        "output_file": f"{model_id.replace('/', '_')}.json",
        "max_tokens": MAX_TOKENS,
        "is_reasoning": False,
    }

def parse_json(content: str, is_reasoning: bool = False) -> dict | None:
    # Reasoning models may include thinking tags
    if is_reasoning:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        content = re.sub(r"<\|reasoning\|>.*?<\|/reasoning\|>", "", content, flags=re.DOTALL).strip()
    # Extract JSON from inside code fences first (ignores trailing text)
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: strip fences and try raw parse
    content = re.sub(r"```(?:json)?|```", "", content).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None

def validate_turn1(data): return data is not None and {"initial_creativity", "enjoyment"}.issubset(data.keys())
def validate_turn2(data):
    req = {"emotional_impact","topic_fidelity","vocabulary_freshness","plot_uniqueness","surprise","empathy","thought_provocation","engagement","stylistic_quality","logical_coherence","tone_fidelity"}
    return data is not None and req.issubset(data.keys())
def validate_turn3(data): return data is not None and "reflective_creativity" in data

def load_completed(f):
    if not f.exists(): return set()
    try: return {r["story_id"] for r in json.load(open(f)).get("results",[]) if r.get("parse_ok")}
    except: return set()

def load_existing_results(f):
    if not f.exists(): return []
    try: return json.load(open(f)).get("results",[])
    except: return []

def save_progress(output_file, results, errors, model_name, label):
    with open(output_file, "w") as f:
        json.dump({
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model": model_name,
            "label": label,
            "design": "conversational-3turn",
            "condition": "unanchored",
            "total_stories": len(STORIES),
            "completed": sum(1 for r in results if r.get("parse_ok")),
            "errors": len(errors),
            "results": results,
        }, f, indent=2)

def run_conversation(client, model_name, story, max_tokens, is_reasoning):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    latencies, raws, parsed = {}, {}, {}

    for turn_n, (turn_key, user_content) in enumerate([
        ("turn1", build_turn1_prompt(story)),
        ("turn2", build_turn2_prompt()),
        ("turn3", build_turn3_prompt()),
    ], 1):
        messages.append({"role": "user", "content": user_content})
        t0 = time.time()
        extra = {"repetition_penalty": 1.15} if is_reasoning else {}
        resp = client.chat.completions.create(
            model=model_name, temperature=TEMPERATURE, max_tokens=max_tokens, messages=messages, extra_body=extra)
        latencies[turn_key] = round((time.time()-t0)*1000)
        raws[turn_key] = resp.choices[0].message.content
        parsed[turn_key] = parse_json(raws[turn_key], is_reasoning)

        validator = [validate_turn1, validate_turn2, validate_turn3][turn_n-1]
        if not validator(parsed[turn_key]):
            return latencies, raws, parsed, False

        messages.append({"role": "assistant", "content": raws[turn_key]})

    return latencies, raws, parsed, True

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    model_name, config = detect_model()
    label = config["label"]
    output_file = OUTPUT_DIR / config["output_file"]
    max_tokens = config["max_tokens"]
    is_reasoning = config["is_reasoning"]

    client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
    completed = load_completed(output_file)
    results = list(load_existing_results(output_file))
    errors = []
    n = len(completed)

    print(f"EXPERIMENT: {model_name} ({label}) | Conversational 3-turn | Unanchored")
    print(f"Reasoning: {'ON' if is_reasoning else 'OFF'} | Max tokens: {max_tokens}")
    print(f"12 stories × 3 turns = 36 API calls")
    if completed: print(f"Resuming: {len(completed)}/12 completed")
    print()

    for story in STORIES:
        if story["id"] in completed: continue
        n += 1
        print(f"[{n:02d}/12] {story['id']}", end=" ", flush=True)
        try:
            lat, raws, par, ok = run_conversation(client, model_name, story, max_tokens, is_reasoning)
            if ok:
                ic, rc = par["turn1"]["initial_creativity"], par["turn3"]["reflective_creativity"]
                flip = rc - ic
                scores = {"initial_creativity": ic, "enjoyment": par["turn1"]["enjoyment"], "sub_components": par["turn2"], "reflective_creativity": rc}
                print(f"IC={ic} RC={rc} flip={flip:+d} ({lat['turn1']}+{lat['turn2']}+{lat['turn3']}ms)")
            else:
                scores, flip = None, None
                print(f"PARSE ERROR")
        except Exception as e:
            lat, raws, scores, flip, ok = {"turn1":0,"turn2":0,"turn3":0}, {}, None, None, False
            print(f"ERROR: {e}")
            errors.append({"story": story["id"], "error": str(e)})

        results.append({"model": model_name, "story_id": story["id"], "topic": story["topic"], "tone": story["tone"], "anchor": "unanchored", "latencies_ms": lat, "raw_responses": raws, "scores": scores, "gk_flip": flip, "parse_ok": ok})
        save_progress(output_file, results, errors, model_name, label)

    pr = [r for r in results if r.get("scores")]
    print(f"\n{'='*60}")
    print(f"Done. {len(pr)}/12 parsed | {len(errors)} errors | Saved -> {output_file}")
    if pr:
        ics = [r["scores"]["initial_creativity"] for r in pr]
        rcs = [r["scores"]["reflective_creativity"] for r in pr]
        flips = [r["gk_flip"] for r in pr if r["gk_flip"] is not None]
        print(f"Means: IC={sum(ics)/len(ics):.2f}  RC={sum(rcs)/len(rcs):.2f}")
        if flips:
            nf = sum(1 for f in flips if f!=0)
            print(f"Flips: {nf}/{len(flips)} mean={sum(flips)/len(flips):+.2f}")
        for r in pr:
            s = r["scores"]
            print(f"  {r['story_id']:30s}: IC={s['initial_creativity']} RC={s['reflective_creativity']} flip={r['gk_flip']:+d}")

if __name__ == "__main__":
    main()
