"""
conv_qwen3_32b.py — Conversational 3-turn experiment: Qwen 3 32B

Runs TWICE: thinking ON and thinking OFF.
12 stories × 2 modes = 24 combos (72 API calls).

Run:
    python conv_qwen3_32b.py

vLLM: python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-32B --tensor-parallel-size 2
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

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "Qwen/Qwen3-32B"
TEMPERATURE = 0.0
MAX_TOKENS_THINKING = 8192
MAX_TOKENS_NOTHINKING = 1024
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "conv_results"

THINKING_MODES = {
    "thinking_on": {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        "max_tokens": MAX_TOKENS_THINKING,
        "output_file": OUTPUT_DIR / "qwen3_32b_thinking_on.json",
    },
    "thinking_off": {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "max_tokens": MAX_TOKENS_NOTHINKING,
        "output_file": OUTPUT_DIR / "qwen3_32b_thinking_off.json",
    },
}

def parse_json(content: str) -> dict | None:
    content = re.sub(r"```(?:json)?|```", "", content).strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
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

def save_progress(output_file, results, errors, mode_label):
    with open(output_file, "w") as f:
        json.dump({"run_id": datetime.now().strftime("%Y%m%d_%H%M%S"), "model": MODEL_NAME, "label": f"Qwen 3 32B ({mode_label})", "design": "conversational-3turn", "condition": "unanchored", "thinking_mode": mode_label, "total_stories": len(STORIES), "completed": sum(1 for r in results if r.get("parse_ok")), "errors": len(errors), "results": results}, f, indent=2)

def run_conversation(client, story, extra_body, max_tokens):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    latencies, raws, parsed = {}, {}, {}

    for turn_n, (turn_key, user_content) in enumerate([
        ("turn1", build_turn1_prompt(story)),
        ("turn2", build_turn2_prompt()),
        ("turn3", build_turn3_prompt()),
    ], 1):
        messages.append({"role": "user", "content": user_content})
        t0 = time.time()
        resp = client.chat.completions.create(
            model=MODEL_NAME, temperature=TEMPERATURE, max_tokens=max_tokens,
            messages=messages, extra_body=extra_body,
        )
        latencies[turn_key] = round((time.time()-t0)*1000)
        raws[turn_key] = resp.choices[0].message.content
        parsed[turn_key] = parse_json(raws[turn_key])

        validator = [validate_turn1, validate_turn2, validate_turn3][turn_n-1]
        if not validator(parsed[turn_key]):
            return latencies, raws, parsed, False

        messages.append({"role": "assistant", "content": raws[turn_key]})

    return latencies, raws, parsed, True

def run_mode(mode_name, mode_config):
    OUTPUT_DIR.mkdir(exist_ok=True)
    client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
    output_file = mode_config["output_file"]

    completed = load_completed(output_file)
    results = list(load_existing_results(output_file))
    errors = []
    n = len(completed)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {MODEL_NAME} | {mode_name} | Conversational 3-turn")
    print(f"12 stories × 3 turns = 36 API calls")
    if completed: print(f"Resuming: {len(completed)}/12 completed")
    print()

    for story in STORIES:
        if story["id"] in completed: continue
        n += 1
        print(f"[{n:02d}/12] {story['id']}", end=" ", flush=True)
        try:
            lat, raws, par, ok = run_conversation(
                client, story, mode_config["extra_body"], mode_config["max_tokens"])
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

        results.append({"model": MODEL_NAME, "thinking_mode": mode_name, "story_id": story["id"], "topic": story["topic"], "tone": story["tone"], "anchor": "unanchored", "latencies_ms": lat, "raw_responses": raws, "scores": scores, "gk_flip": flip, "parse_ok": ok})
        save_progress(output_file, results, errors, mode_name)

    pr = [r for r in results if r.get("scores")]
    print(f"\nDone. {len(pr)}/12 parsed | {len(errors)} errors | Saved -> {output_file}")
    if pr:
        ics = [r["scores"]["initial_creativity"] for r in pr]
        rcs = [r["scores"]["reflective_creativity"] for r in pr]
        flips = [r["gk_flip"] for r in pr if r["gk_flip"] is not None]
        print(f"Means: IC={sum(ics)/len(ics):.2f}  RC={sum(rcs)/len(rcs):.2f}")
        if flips:
            nf = sum(1 for f in flips if f!=0)
            print(f"Flips: {nf}/{len(flips)} mean={sum(flips)/len(flips):+.2f}")

def main():
    for mode_name, mode_config in THINKING_MODES.items():
        run_mode(mode_name, mode_config)
    print(f"\n{'='*60}")
    print("All modes complete.")

if __name__ == "__main__":
    main()
