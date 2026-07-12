"""
conv_llama_70b.py — Conversational 3-turn experiment: Llama 3.1 70B Instruct

12 stories × 1 condition = 12 combos (36 API calls).
No stance, no reasoning toggle.

Run:
    python conv_llama_70b.py

vLLM: --tensor-parallel-size 2 (or more) for 70B.
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
MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"
TEMPERATURE = 0.0
MAX_TOKENS = 1024
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "conv_results"
OUTPUT_FILE = OUTPUT_DIR / "llama31_70b.json"

def parse_json(content: str) -> dict | None:
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

def save_progress(output_file, results, errors):
    with open(output_file, "w") as f:
        json.dump({"run_id": datetime.now().strftime("%Y%m%d_%H%M%S"), "model": MODEL_NAME, "label": "Llama 3.1 70B", "design": "conversational-3turn", "condition": "unanchored", "total_stories": len(STORIES), "completed": sum(1 for r in results if r.get("parse_ok")), "errors": len(errors), "results": results}, f, indent=2)

def run_conversation(client, story):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    latencies, raws, parsed = {}, {}, {}

    messages.append({"role": "user", "content": build_turn1_prompt(story)})
    t0 = time.time()
    resp = client.chat.completions.create(model=MODEL_NAME, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, messages=messages)
    latencies["turn1"] = round((time.time()-t0)*1000)
    raws["turn1"] = resp.choices[0].message.content
    parsed["turn1"] = parse_json(raws["turn1"])
    if not validate_turn1(parsed["turn1"]): return latencies, raws, parsed, False
    messages.append({"role": "assistant", "content": raws["turn1"]})

    messages.append({"role": "user", "content": build_turn2_prompt()})
    t0 = time.time()
    resp = client.chat.completions.create(model=MODEL_NAME, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, messages=messages)
    latencies["turn2"] = round((time.time()-t0)*1000)
    raws["turn2"] = resp.choices[0].message.content
    parsed["turn2"] = parse_json(raws["turn2"])
    if not validate_turn2(parsed["turn2"]): return latencies, raws, parsed, False
    messages.append({"role": "assistant", "content": raws["turn2"]})

    messages.append({"role": "user", "content": build_turn3_prompt()})
    t0 = time.time()
    resp = client.chat.completions.create(model=MODEL_NAME, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, messages=messages)
    latencies["turn3"] = round((time.time()-t0)*1000)
    raws["turn3"] = resp.choices[0].message.content
    parsed["turn3"] = parse_json(raws["turn3"])
    if not validate_turn3(parsed["turn3"]): return latencies, raws, parsed, False

    return latencies, raws, parsed, True

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
    completed = load_completed(OUTPUT_FILE)
    results = list(load_existing_results(OUTPUT_FILE))
    errors = []
    n = len(completed)

    print(f"EXPERIMENT: {MODEL_NAME} | Conversational 3-turn | Unanchored")
    print(f"12 stories × 3 turns = 36 API calls")
    if completed: print(f"Resuming: {len(completed)}/12 completed")
    print()

    for story in STORIES:
        if story["id"] in completed: continue
        n += 1
        print(f"[{n:02d}/12] {story['id']}", end=" ", flush=True)
        try:
            lat, raws, par, ok = run_conversation(client, story)
            if ok:
                ic, rc = par["turn1"]["initial_creativity"], par["turn3"]["reflective_creativity"]
                flip = rc - ic
                scores = {"initial_creativity": ic, "enjoyment": par["turn1"]["enjoyment"], "sub_components": par["turn2"], "reflective_creativity": rc}
                print(f"IC={ic} RC={rc} flip={flip:+d} ({lat['turn1']}+{lat['turn2']}+{lat['turn3']}ms)")
            else:
                scores, flip = None, None
                print(f"PARSE ERROR")
        except Exception as e:
            lat, raws, par, scores, flip, ok = {"turn1":0,"turn2":0,"turn3":0}, {}, {}, None, None, False
            print(f"ERROR: {e}")
            errors.append({"story": story["id"], "error": str(e)})

        results.append({"model": MODEL_NAME, "story_id": story["id"], "topic": story["topic"], "tone": story["tone"], "anchor": "unanchored", "latencies_ms": lat, "raw_responses": raws, "scores": scores, "gk_flip": flip, "parse_ok": ok})
        save_progress(OUTPUT_FILE, results, errors)

    pr = [r for r in results if r.get("scores")]
    print(f"\n{'='*60}")
    print(f"Done. {len(pr)}/12 parsed | {len(errors)} errors | Saved -> {OUTPUT_FILE}")
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
