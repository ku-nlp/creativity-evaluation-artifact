import json
import os
import time
import re
import argparse
from datetime import datetime
from pathlib import Path
from google import genai
from google.genai import types

# Import shared prompts
from shared_prompts import (
    SYSTEM_PROMPT,
    build_turn1_prompt, build_turn2_prompt, build_turn3_prompt
)

def parse_json(content: str):
    # Strip thinking blocks if present
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
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

def run_conversation(client, model_id, story, thinking_level="none"):
    """Run 3-turn conversation using Gemini's multi-turn chat."""
    thinking_on = (thinking_level != "none")
    gen_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=1.0 if thinking_on else 0.7,
        max_output_tokens=16384 if thinking_on else 4096,
    )
    
    if thinking_on:
        # Use budget for all models as level is not supported in this SDK version
        budget_map = {"low": 4096, "medium": 16384, "high": 32768}
        gen_config.thinking_config = types.ThinkingConfig(
            thinking_budget=budget_map.get(thinking_level, 16384),
            include_thoughts=True
        )

    chat = client.chats.create(
        model=model_id,
        config=gen_config,
    )

    latencies, raws, parsed = {}, {}, {}

    # -- Turn 1 --
    t0 = time.time()
    try:
        response = chat.send_message(build_turn1_prompt(story))
        latencies["turn1"] = round((time.time() - t0) * 1000)
        raws["turn1"] = response.text or ""
        parsed["turn1"] = parse_json(raws["turn1"])
    except Exception as e:
        print(f"Turn 1 error: {e}")
        return latencies, raws, parsed, False
    
    if not validate_turn1(parsed["turn1"]): return latencies, raws, parsed, False

    # -- Turn 2 --
    t0 = time.time()
    try:
        response = chat.send_message(build_turn2_prompt())
        latencies["turn2"] = round((time.time() - t0) * 1000)
        raws["turn2"] = response.text or ""
        parsed["turn2"] = parse_json(raws["turn2"])
    except Exception as e:
        print(f"Turn 2 error: {e}")
        return latencies, raws, parsed, False

    if not validate_turn2(parsed["turn2"]): return latencies, raws, parsed, False

    # -- Turn 3 --
    t0 = time.time()
    try:
        response = chat.send_message(build_turn3_prompt())
        latencies["turn3"] = round((time.time() - t0) * 1000)
        raws["turn3"] = response.text or ""
        parsed["turn3"] = parse_json(raws["turn3"])
    except Exception as e:
        print(f"Turn 3 error: {e}")
        return latencies, raws, parsed, False

    if not validate_turn3(parsed["turn3"]): return latencies, raws, parsed, False

    return latencies, raws, parsed, True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="gemini-2.5-pro, gemini-3.1-pro-preview, etc.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--stories", default="stories_80.json")
    parser.add_argument("--thinking-level", choices=["none", "low", "medium", "high"], default="none")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.")
        return

    client = genai.Client(api_key=api_key)
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    stories_path = BASE_DIR / "data" / args.stories
    with open(stories_path, "r") as f:
        stories = json.load(f)

    OUTPUT_DIR = BASE_DIR / "data" / "raw_json"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    run_label = f"{args.label}_{args.thinking_level}"
    OUTPUT_FILE = OUTPUT_DIR / f"results_{run_label}.json"

    results = []
    if OUTPUT_FILE.exists():
        try:
            results = json.load(open(OUTPUT_FILE)).get("results", [])
        except:
            results = []

    completed_ids = {r["story_id"] for r in results if r.get("parse_ok")}

    print(f"\n{'='*60}")
    print(f"JUDGE: {args.model} | LEVEL: {args.thinking_level} | Stories: {args.stories}")
    print(f"Progress: {len(completed_ids)}/{len(stories)}")
    print(f"{'='*60}")

    for i, story in enumerate(stories, 1):
        if story["id"] in completed_ids: continue

        print(f"[{i:02d}/80] {story['id']}...", end=" ", flush=True)
        try:
            latencies, raws, parsed, ok = run_conversation(client, args.model, story, args.thinking_level)
            if ok:
                ic = parsed["turn1"]["initial_creativity"]
                rc = parsed["turn3"]["reflective_creativity"]
                print(f"IC={ic} RC={rc} ({latencies['turn1']+latencies['turn2']+latencies['turn3']}ms)")
                results.append({
                    "model": args.model,
                    "thinking_mode": (args.thinking_level != "none"),
                    "thinking_level": args.thinking_level,
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
                    "thinking_mode": (args.thinking_level != "none"),
                    "thinking_level": args.thinking_level,
                    "parse_ok": False,
                    "raw_responses": raws
                })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "story_id": story["id"],
                "thinking_mode": (args.thinking_level != "none"),
                "thinking_level": args.thinking_level,
                "parse_ok": False,
                "error": str(e)
            })

        with open(OUTPUT_FILE, "w") as f:
            json.dump({"model": args.model, "thinking_level": args.thinking_level, "results": results}, f, indent=2)

if __name__ == "__main__":
    main()
