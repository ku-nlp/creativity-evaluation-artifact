"""
conv_llama_8b.py — Conversational 3-turn experiment: Llama 3.1 8B Instruct

12 stories × 1 condition = 12 combos (36 API calls).
No stance, no reasoning toggle.

Run:
    python conv_llama_8b.py
"""

import json
import time
import re
from datetime import datetime
from pathlib import Path
from openai import OpenAI

from shared_stories import (
    STORIES, SYSTEM_PROMPT,
    build_turn1_prompt, build_turn2_prompt, build_turn3_prompt,
)

# -- CONFIG ---------------------------------------------------------------

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TEMPERATURE = 0.0
MAX_TOKENS = 1024
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "conv_results"
OUTPUT_FILE = OUTPUT_DIR / "llama31_8b.json"

# -- PARSING ---------------------------------------------------------------

def parse_json(content: str) -> dict | None:
    content = re.sub(r"```(?:json)?|```", "", content).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None

def validate_turn1(data: dict) -> bool:
    return data is not None and {"initial_creativity", "enjoyment"}.issubset(data.keys())

def validate_turn2(data: dict) -> bool:
    required = {"emotional_impact", "topic_fidelity", "vocabulary_freshness", "plot_uniqueness",
                "surprise", "empathy", "thought_provocation", "engagement",
                "stylistic_quality", "logical_coherence", "tone_fidelity"}
    return data is not None and required.issubset(data.keys())

def validate_turn3(data: dict) -> bool:
    return data is not None and "reflective_creativity" in data

# -- RESUME ----------------------------------------------------------------

def load_completed(output_file: Path) -> set:
    if not output_file.exists():
        return set()
    try:
        data = json.load(open(output_file))
        return {r["story_id"] for r in data.get("results", []) if r.get("parse_ok")}
    except (json.JSONDecodeError, KeyError):
        return set()

def load_existing_results(output_file: Path) -> list:
    if not output_file.exists():
        return []
    try:
        return json.load(open(output_file)).get("results", [])
    except (json.JSONDecodeError, KeyError):
        return []

def save_progress(output_file: Path, results: list, errors: list, model: str, label: str):
    output = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model": model,
        "label": label,
        "design": "conversational-3turn",
        "condition": "unanchored",
        "total_stories": len(STORIES),
        "completed": sum(1 for r in results if r.get("parse_ok")),
        "errors": len(errors),
        "results": results,
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

# -- CONVERSATIONAL RUN ----------------------------------------------------

def run_conversation(client, model_name, story, extra_body=None):
    """Run 3-turn conversational evaluation for one story."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    latencies = {}
    raws = {}
    parsed = {}

    # -- Turn 1: IC + Enjoyment --
    messages.append({"role": "user", "content": build_turn1_prompt(story)})
    t0 = time.time()
    kwargs = dict(model=model_name, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, messages=messages)
    if extra_body:
        kwargs["extra_body"] = extra_body
    resp1 = client.chat.completions.create(**kwargs)
    latencies["turn1"] = round((time.time() - t0) * 1000)
    raws["turn1"] = resp1.choices[0].message.content
    parsed["turn1"] = parse_json(raws["turn1"])

    if not validate_turn1(parsed["turn1"]):
        return latencies, raws, parsed, False

    # Append assistant response to conversation
    messages.append({"role": "assistant", "content": raws["turn1"]})

    # -- Turn 2: 11 Sub-components --
    messages.append({"role": "user", "content": build_turn2_prompt()})
    t0 = time.time()
    kwargs["messages"] = messages
    resp2 = client.chat.completions.create(**kwargs)
    latencies["turn2"] = round((time.time() - t0) * 1000)
    raws["turn2"] = resp2.choices[0].message.content
    parsed["turn2"] = parse_json(raws["turn2"])

    if not validate_turn2(parsed["turn2"]):
        return latencies, raws, parsed, False

    messages.append({"role": "assistant", "content": raws["turn2"]})

    # -- Turn 3: Reflective Creativity --
    messages.append({"role": "user", "content": build_turn3_prompt()})
    t0 = time.time()
    kwargs["messages"] = messages
    resp3 = client.chat.completions.create(**kwargs)
    latencies["turn3"] = round((time.time() - t0) * 1000)
    raws["turn3"] = resp3.choices[0].message.content
    parsed["turn3"] = parse_json(raws["turn3"])

    if not validate_turn3(parsed["turn3"]):
        return latencies, raws, parsed, False

    return latencies, raws, parsed, True

# -- MAIN ------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")

    completed = load_completed(OUTPUT_FILE)
    results = list(load_existing_results(OUTPUT_FILE))
    errors = []
    n = len(completed)

    print(f"EXPERIMENT: {MODEL_NAME} | Conversational 3-turn | Unanchored")
    print(f"12 stories × 3 turns = 36 API calls")
    if completed:
        print(f"Resuming: {len(completed)}/12 already completed")
    print()

    for story in STORIES:
        if story["id"] in completed:
            continue

        n += 1
        label = f"[{n:02d}/12] {story['id']}"
        print(f"{label}", end=" ", flush=True)

        try:
            latencies, raws, parsed, ok = run_conversation(client, MODEL_NAME, story)

            if ok:
                ic = parsed["turn1"]["initial_creativity"]
                rc = parsed["turn3"]["reflective_creativity"]
                flip = rc - ic
                scores = {
                    "initial_creativity": ic,
                    "enjoyment": parsed["turn1"]["enjoyment"],
                    "sub_components": parsed["turn2"],
                    "reflective_creativity": rc,
                }
                print(f"IC={ic} RC={rc} flip={flip:+d} ({latencies['turn1']}+{latencies['turn2']}+{latencies['turn3']}ms)")
            else:
                scores, flip = None, None
                failed = [k for k, v in parsed.items() if v is None or
                          (k == "turn1" and not validate_turn1(v)) or
                          (k == "turn2" and not validate_turn2(v)) or
                          (k == "turn3" and not validate_turn3(v))]
                print(f"PARSE ERROR at {failed}")

        except Exception as e:
            latencies = {"turn1": 0, "turn2": 0, "turn3": 0}
            raws = {"turn1": str(e), "turn2": "", "turn3": ""}
            parsed = {"turn1": None, "turn2": None, "turn3": None}
            scores, flip, ok = None, None, False
            print(f"ERROR: {e}")
            errors.append({"story": story["id"], "error": str(e)})

        results.append({
            "model": MODEL_NAME,
            "story_id": story["id"],
            "topic": story["topic"],
            "tone": story["tone"],
            "anchor": "unanchored",
            "latencies_ms": latencies,
            "raw_responses": raws,
            "scores": scores,
            "gk_flip": flip,
            "parse_ok": ok,
        })

        save_progress(OUTPUT_FILE, results, errors, MODEL_NAME, "Llama 3.1 8B")

    # -- Summary --
    parsed_results = [r for r in results if r.get("scores")]
    print(f"\n{'='*60}")
    print(f"Done. {len(parsed_results)}/12 parsed | {len(errors)} errors")
    print(f"Saved -> {OUTPUT_FILE}")

    if not parsed_results:
        return

    ics = [r["scores"]["initial_creativity"] for r in parsed_results]
    rcs = [r["scores"]["reflective_creativity"] for r in parsed_results]
    enjs = [r["scores"]["enjoyment"] for r in parsed_results]
    flips = [r["gk_flip"] for r in parsed_results if r["gk_flip"] is not None]

    print(f"\nMeans: IC={sum(ics)/len(ics):.2f}  RC={sum(rcs)/len(rcs):.2f}  Enj={sum(enjs)/len(enjs):.2f}")

    if flips:
        nf = sum(1 for f in flips if f != 0)
        print(f"Flips: {nf}/{len(flips)} ({100*nf/len(flips):.0f}%)  mean={sum(flips)/len(flips):+.2f}")

    print(f"\nPer-story:")
    for r in parsed_results:
        s = r["scores"]
        print(f"  {r['story_id']:30s}: IC={s['initial_creativity']} RC={s['reflective_creativity']} Enj={s['enjoyment']} flip={r['gk_flip']:+d}")

if __name__ == "__main__":
    main()
