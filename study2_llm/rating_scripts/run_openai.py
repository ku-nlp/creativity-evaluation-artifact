import json
import os
import time
import re
import argparse
from datetime import datetime
from pathlib import Path
from openai import OpenAI

# Import shared prompts
from shared_prompts import (
    SYSTEM_PROMPT,
    build_turn1_prompt, build_turn2_prompt, build_turn3_prompt
)

def parse_json(content: str):
    # Strip thinking blocks if present
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass
    content = re.sub(r"```(?:json)?|```", "", content).strip()
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

TURN1_KEYS = ["initial_creativity", "enjoyment"]
TURN2_KEYS = ["emotional_impact", "topic_fidelity", "vocabulary_freshness", "plot_uniqueness",
              "surprise", "empathy", "thought_provocation", "engagement",
              "stylistic_quality", "logical_coherence", "tone_fidelity"]

def validate_turn1(data):
    return data is not None and set(TURN1_KEYS).issubset(data.keys())

def validate_turn2(data):
    return data is not None and set(TURN2_KEYS).issubset(data.keys())

def validate_turn3(data):
    return data is not None and "reflective_creativity" in data

def run_conversation(client, model_id, story, thinking_on=False):
    """Run 3-turn conversation using OpenAI's reasoning_effort."""
    kwargs = {
        "model": model_id,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}]
    }
    
    # GPT-5.5 reasoning_effort parameter
    if thinking_on:
        kwargs["reasoning_effort"] = "high"
    else:
        # For standard turbo models, we can just use low or omit
        kwargs["reasoning_effort"] = "none"

    latencies, raws, parsed = {}, {}, {}
    messages = list(kwargs["messages"])

    # -- Turn 1 --
    messages.append({"role": "user", "content": build_turn1_prompt(story)})
    t0 = time.time()
    response = client.chat.completions.create(**{**kwargs, "messages": messages})
    latencies["turn1"] = round((time.time() - t0) * 1000)
    raw1 = response.choices[0].message.content or ""
    raws["turn1"] = raw1
    parsed["turn1"] = parse_json(raw1)
    if not validate_turn1(parsed["turn1"]): return latencies, raws, parsed, False
    messages.append({"role": "assistant", "content": json.dumps(parsed["turn1"])})

    # -- Turn 2 --
    messages.append({"role": "user", "content": build_turn2_prompt()})
    t0 = time.time()
    response = client.chat.completions.create(**{**kwargs, "messages": messages})
    latencies["turn2"] = round((time.time() - t0) * 1000)
    raw2 = response.choices[0].message.content or ""
    raws["turn2"] = raw2
    parsed["turn2"] = parse_json(raw2)
    if not validate_turn2(parsed["turn2"]): return latencies, raws, parsed, False
    messages.append({"role": "assistant", "content": json.dumps(parsed["turn2"])})

    # -- Turn 3 --
    messages.append({"role": "user", "content": build_turn3_prompt()})
    t0 = time.time()
    response = client.chat.completions.create(**{**kwargs, "messages": messages})
    latencies["turn3"] = round((time.time() - t0) * 1000)
    raw3 = response.choices[0].message.content or ""
    raws["turn3"] = raw3
    parsed["turn3"] = parse_json(raw3)
    if not validate_turn3(parsed["turn3"]): return latencies, raws, parsed, False

    return latencies, raws, parsed, True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.5-turbo")
    parser.add_argument("--label", default="gpt55_turbo")
    parser.add_argument("--stories", default="stories_80.json")
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"], default="none")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        return

    client = OpenAI(api_key=api_key)
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    stories_path = BASE_DIR / "data" / args.stories
    with open(stories_path, "r") as f:
        stories = json.load(f)

    OUTPUT_DIR = BASE_DIR / "data" / "raw_json"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    thinking_on = (args.reasoning_effort != "none")
    run_label = f"{args.label}_{args.reasoning_effort}"
    OUTPUT_FILE = OUTPUT_DIR / f"results_{run_label}.json"

    results = []
    if OUTPUT_FILE.exists():
        results = json.load(open(OUTPUT_FILE)).get("results", [])

    completed_ids = {r["story_id"] for r in results if r.get("parse_ok")}

    print(f"\n{'='*60}")
    print(f"JUDGE: {args.model} | EFFORT: {args.reasoning_effort} | Stories: {args.stories}")
    print(f"Progress: {len(completed_ids)}/{len(stories)}")
    print(f"{'='*60}")

    for i, story in enumerate(stories, 1):
        if story["id"] in completed_ids: continue

        print(f"[{i:02d}/80] {story['id']}...", end=" ", flush=True)
        try:
            latencies, raws, parsed, ok = run_conversation(client, args.model, story, args.reasoning_effort)
            if ok:
                ic = parsed["turn1"]["initial_creativity"]
                rc = parsed["turn3"]["reflective_creativity"]
                print(f"IC={ic} RC={rc} ({latencies['turn1']+latencies['turn2']+latencies['turn3']}ms)")
                results.append({
                    "model": args.model,
                    "thinking_mode": thinking_on,
                    "story_id": story["id"],
                    "latencies_ms": latencies,
                    "raw_responses": raws,
                    "scores": {
                        "initial_creativity": ic,
                        "enjoyment": parsed["turn1"]["enjoyment"],
                        "sub_components": parsed["turn2"],
                        "reflective_creativity": rc,
                    },
                    "gk_flip": rc - ic,
                    "parse_ok": True
                })
            else:
                print("PARSE FAILED")
                results.append({
                    "story_id": story["id"],
                    "thinking_mode": thinking_on,
                    "parse_ok": False,
                    "raw_responses": raws
                })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "story_id": story["id"],
                "thinking_mode": thinking_on,
                "parse_ok": False,
                "error": str(e)
            })

        with open(OUTPUT_FILE, "w") as f:
            json.dump({"model": args.model, "thinking_mode": thinking_on, "results": results}, f, indent=2)

if __name__ == "__main__":
    main()
