"""
conv_openai.py — Conversational 3-turn experiment: GPT-4.1 + o4-mini

GPT-4.1: standard model (no reasoning)
o4-mini: reasoning model

12 stories x 2 models = 24 combos (72 API calls).

Run:
    python conv_openai.py

Requires OPENAI_API_KEY in environment or .env file.
"""

import json
import os
import time
import re
from datetime import datetime
from pathlib import Path
from openai import OpenAI

from shared_stories import (
    STORIES, SYSTEM_PROMPT,
    build_turn1_prompt, build_turn2_prompt, build_turn3_prompt,
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TEMPERATURE = 0.0
MAX_TOKENS = 1024
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "conv_results"

MODELS = {
    "gpt-4.1": {
        "model_id": "gpt-4.1",
        "label": "GPT-4.1",
        "output_file": "gpt41.json",
        "is_reasoning": False,
    },
    "o4-mini": {
        "model_id": "o4-mini",
        "label": "o4-mini (reasoning)",
        "output_file": "o4_mini.json",
        "is_reasoning": True,
    },
}


def parse_json(content: str):
    # Extract from code fences first
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: strip fences, try raw parse
    content = re.sub(r"```(?:json)?|```", "", content).strip()
    # Try to extract first JSON object
    brace_match = re.search(r"\{[^{}]*\}", content)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def validate_turn1(data):
    return data is not None and {"initial_creativity", "enjoyment"}.issubset(data.keys())

def validate_turn2(data):
    req = {"emotional_impact", "topic_fidelity", "vocabulary_freshness", "plot_uniqueness",
           "surprise", "empathy", "thought_provocation", "engagement", "stylistic_quality",
           "logical_coherence", "tone_fidelity"}
    return data is not None and req.issubset(data.keys())

def validate_turn3(data):
    return data is not None and "reflective_creativity" in data


def load_completed(f):
    if not f.exists():
        return set()
    try:
        return {r["story_id"] for r in json.load(open(f)).get("results", []) if r.get("parse_ok")}
    except Exception:
        return set()

def load_existing_results(f):
    if not f.exists():
        return []
    try:
        return json.load(open(f)).get("results", [])
    except Exception:
        return []

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


def run_conversation(client, model_id, story, is_reasoning):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    latencies, raws, parsed = {}, {}, {}

    for turn_n, (turn_key, user_content) in enumerate([
        ("turn1", build_turn1_prompt(story)),
        ("turn2", build_turn2_prompt()),
        ("turn3", build_turn3_prompt()),
    ], 1):
        messages.append({"role": "user", "content": user_content})
        t0 = time.time()

        if is_reasoning:
            # o-series models: no temperature/system prompt via normal params
            # Use max_completion_tokens instead of max_tokens
            resp = client.chat.completions.create(
                model=model_id,
                max_completion_tokens=MAX_TOKENS * 8,
                messages=messages,
            )
        else:
            resp = client.chat.completions.create(
                model=model_id,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=messages,
            )

        latencies[turn_key] = round((time.time() - t0) * 1000)
        raws[turn_key] = resp.choices[0].message.content
        parsed[turn_key] = parse_json(raws[turn_key])

        validator = [validate_turn1, validate_turn2, validate_turn3][turn_n - 1]
        if not validator(parsed[turn_key]):
            return latencies, raws, parsed, False

        messages.append({"role": "assistant", "content": raws[turn_key]})

    return latencies, raws, parsed, True


def run_model(model_name, config):
    OUTPUT_DIR.mkdir(exist_ok=True)
    client = OpenAI(api_key=OPENAI_API_KEY)
    output_file = OUTPUT_DIR / config["output_file"]
    model_id = config["model_id"]
    label = config["label"]
    is_reasoning = config["is_reasoning"]

    completed = load_completed(output_file)
    results = list(load_existing_results(output_file))
    errors = []
    n = len(completed)

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {model_id} ({label}) | Conversational 3-turn | Unanchored")
    print(f"Reasoning: {'ON' if is_reasoning else 'OFF'}")
    print(f"12 stories x 3 turns = 36 API calls")
    if completed:
        print(f"Resuming: {len(completed)}/12 completed")
    print()

    for story in STORIES:
        if story["id"] in completed:
            continue
        n += 1
        print(f"[{n:02d}/12] {story['id']}", end=" ", flush=True)
        try:
            lat, raws, par, ok = run_conversation(client, model_id, story, is_reasoning)
            if ok:
                ic = par["turn1"]["initial_creativity"]
                rc = par["turn3"]["reflective_creativity"]
                flip = rc - ic
                scores = {
                    "initial_creativity": ic,
                    "enjoyment": par["turn1"]["enjoyment"],
                    "sub_components": par["turn2"],
                    "reflective_creativity": rc,
                }
                print(f"IC={ic} RC={rc} flip={flip:+d} ({lat['turn1']}+{lat['turn2']}+{lat['turn3']}ms)")
            else:
                scores, flip = None, None
                print("PARSE ERROR")
        except Exception as e:
            lat = {"turn1": 0, "turn2": 0, "turn3": 0}
            raws, scores, flip, ok = {}, None, None, False
            print(f"ERROR: {e}")
            errors.append({"story": story["id"], "error": str(e)})

        results.append({
            "model": model_id,
            "story_id": story["id"],
            "topic": story["topic"],
            "tone": story["tone"],
            "anchor": "unanchored",
            "latencies_ms": lat,
            "raw_responses": raws,
            "scores": scores,
            "gk_flip": flip,
            "parse_ok": ok,
        })
        save_progress(output_file, results, errors, model_id, label)

    pr = [r for r in results if r.get("scores")]
    print(f"\nDone. {len(pr)}/12 parsed | {len(errors)} errors | Saved -> {output_file}")
    if pr:
        ics = [r["scores"]["initial_creativity"] for r in pr]
        rcs = [r["scores"]["reflective_creativity"] for r in pr]
        flips = [r["gk_flip"] for r in pr if r["gk_flip"] is not None]
        print(f"Means: IC={sum(ics)/len(ics):.2f}  RC={sum(rcs)/len(rcs):.2f}")
        if flips:
            nf = sum(1 for f in flips if f != 0)
            print(f"Flips: {nf}/{len(flips)} mean={sum(flips)/len(flips):+.2f}")
        for r in pr:
            s = r["scores"]
            print(f"  {r['story_id']:30s}: IC={s['initial_creativity']} RC={s['reflective_creativity']} flip={r['gk_flip']:+d}")


def main():
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set. Export it or add to .env")
        return

    for model_name, config in MODELS.items():
        run_model(model_name, config)

    print(f"\n{'=' * 60}")
    print("All OpenAI models complete.")


if __name__ == "__main__":
    main()
