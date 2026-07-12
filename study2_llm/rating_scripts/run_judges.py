import os
import re
import json
import time
import argparse
from pathlib import Path
from openai import OpenAI

# Import shared prompts
from shared_prompts import (
    SYSTEM_PROMPT,
    build_turn1_prompt, build_turn2_prompt, build_turn3_prompt
)

def strip_think_block(text: str) -> str:
    """Remove <think>...</think> blocks and common 'Thinking Process' headers."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    # Also remove "Thinking Process" or "Thought Process" style headers
    text = re.sub(r"^(?:Thinking|Thought)\s+Process:?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()


def parse_json(content: str) -> dict | None:
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


def _values_in_range(data: dict, keys: list, lo: int = 1, hi: int = 7) -> bool:
    """Check that all values for the given keys are integers in [lo, hi]."""
    for k in keys:
        v = data.get(k)
        if not isinstance(v, (int, float)) or v < lo or v > hi:
            return False
    return True


def validate_turn1(data: dict) -> bool:
    return (data is not None
            and set(TURN1_KEYS).issubset(data.keys())
            and _values_in_range(data, TURN1_KEYS))

def validate_turn2(data: dict) -> bool:
    return (data is not None
            and set(TURN2_KEYS).issubset(data.keys())
            and _values_in_range(data, TURN2_KEYS))

def validate_turn3(data: dict) -> bool:
    return (data is not None
            and "reflective_creativity" in data
            and _values_in_range(data, ["reflective_creativity"]))

def _make_request(client, model_name, messages, thinking_on, no_extra_body=False,
                  temperature=None):
    """Single API call. thinking_on controls whether Qwen thinks or not.
    temperature=None keeps the per-mode study defaults below; a float
    overrides temperature only (all other sampling params stay at study
    values), used by temp_sensitivity/run_vllm.py sweeps."""
    # Use much larger max_tokens because Qwen 122B is long-winded even in nothink mode
    is_122b = "122b" in model_name.lower()
    kwargs = dict(
        model=model_name,
        messages=messages,
        max_tokens=16384 if is_122b else (8192 if thinking_on else 4096),
    )
    if no_extra_body:
        # Standard params for non-Qwen models (Llama etc.)
        kwargs["temperature"] = 0.7
        kwargs["top_p"] = 0.9
    elif thinking_on:
        # Qwen3.5 official recommended params for thinking mode (General Task)
        kwargs["temperature"] = 1.0
        kwargs["top_p"] = 0.95
        kwargs["presence_penalty"] = 1.5
        kwargs["extra_body"] = {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    else:
        # Qwen3.5 official recommended params for non-thinking mode
        kwargs["temperature"] = 0.7
        kwargs["top_p"] = 0.8
        kwargs["presence_penalty"] = 1.5
        kwargs["extra_body"] = {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    if temperature is not None:
        kwargs["temperature"] = temperature
    return client.chat.completions.create(**kwargs)


def run_conversation(client, model_name, story, thinking_on, no_extra_body=False,
                     temperature=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    latencies = {}
    raws = {}
    parsed = {}

    # -- Turn 1 --
    messages.append({"role": "user", "content": build_turn1_prompt(story)})
    t0 = time.time()
    resp1 = _make_request(client, model_name, messages, thinking_on, no_extra_body, temperature)
    latencies["turn1"] = round((time.time() - t0) * 1000)
    raw1 = resp1.choices[0].message.content or ""
    raws["turn1"] = raw1
    parsed["turn1"] = fix_keys(parse_json(raw1), TURN1_KEYS)
    if not validate_turn1(parsed["turn1"]): return latencies, raws, parsed, False
    messages.append({"role": "assistant", "content": json.dumps(parsed["turn1"])})

    # -- Turn 2 --
    messages.append({"role": "user", "content": build_turn2_prompt()})
    t0 = time.time()
    resp2 = _make_request(client, model_name, messages, thinking_on, no_extra_body, temperature)
    latencies["turn2"] = round((time.time() - t0) * 1000)
    raw2 = resp2.choices[0].message.content or ""
    raws["turn2"] = raw2
    parsed["turn2"] = fix_keys(parse_json(raw2), TURN2_KEYS)
    if not validate_turn2(parsed["turn2"]): return latencies, raws, parsed, False
    messages.append({"role": "assistant", "content": json.dumps(parsed["turn2"])})

    # -- Turn 3 --
    messages.append({"role": "user", "content": build_turn3_prompt()})
    t0 = time.time()
    resp3 = _make_request(client, model_name, messages, thinking_on, no_extra_body, temperature)
    latencies["turn3"] = round((time.time() - t0) * 1000)
    raw3 = resp3.choices[0].message.content or ""
    raws["turn3"] = raw3
    parsed["turn3"] = parse_json(raw3)
    if not validate_turn3(parsed["turn3"]): return latencies, raws, parsed, False

    return latencies, raws, parsed, True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--stories", default="stories_80.json")
    parser.add_argument("--mode", choices=["both", "nothink", "think"], default="both",
                        help="Run thinking off, thinking on, or both (default: both)")
    parser.add_argument("--no-extra-body", action="store_true",
                        help="Skip Qwen-specific extra_body params (for Llama etc.)")
    parser.add_argument("--revalidate", action="store_true",
                        help="Re-run stories that have out-of-range scores (>7 or <1)")
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent.parent
    stories_path = Path(args.stories)
    if stories_path.is_absolute():
        INPUT_FILE = stories_path
    elif "/" in args.stories:
        INPUT_FILE = BASE_DIR / args.stories
    else:
        INPUT_FILE = BASE_DIR / "data" / args.stories
    OUTPUT_DIR = BASE_DIR / "data" / "raw_json"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_FILE, "r") as f:
        stories = json.load(f)

    client = OpenAI(base_url=args.url, api_key="not-needed")

    if args.mode == "both":
        modes = [("nothink", False), ("think", True)]
    elif args.mode == "nothink":
        modes = [("nothink", False)]
    else:
        modes = [("think", True)]

    for mode_label, thinking_on in modes:
        run_label = f"{args.label}_{mode_label}"
        OUTPUT_FILE = OUTPUT_DIR / f"results_{run_label}.json"

        results = []
        if OUTPUT_FILE.exists():
            results = json.load(open(OUTPUT_FILE)).get("results", [])

        if args.revalidate:
            # Find stories with out-of-range sub-component scores and remove them
            clean_results = []
            rerun_ids = set()
            for r in results:
                if not r.get("parse_ok"):
                    clean_results.append(r)
                    continue
                subs = r.get("scores", {}).get("sub_components", {})
                has_oor = any(isinstance(v, (int, float)) and (v < 1 or v > 7)
                             for v in subs.values())
                # Also check IC, enjoyment, RC
                for key in ["initial_creativity", "enjoyment", "reflective_creativity"]:
                    v = r.get("scores", {}).get(key)
                    if isinstance(v, (int, float)) and (v < 1 or v > 7):
                        has_oor = True
                if has_oor:
                    rerun_ids.add(r["story_id"])
                else:
                    clean_results.append(r)
            if rerun_ids:
                print(f"  REVALIDATE: {len(rerun_ids)} stories have out-of-range scores, will re-run")
            results = clean_results

        completed_ids = {r["story_id"] for r in results if r.get("parse_ok")}

        think_str = "THINKING ON" if thinking_on else "THINKING OFF"
        print(f"\n{'='*60}")
        print(f"JUDGE: {args.model} | {think_str} | Stories: {args.stories}")
        print(f"Progress: {len(completed_ids)}/{len(stories)} | Output: results_{run_label}.json")
        print(f"{'='*60}")

        MAX_RETRIES = 10

        for i, story in enumerate(stories, 1):
            if story["id"] in completed_ids: continue

            print(f"[{i:02d}/80] {story['id']}...", end=" ", flush=True)
            success = False
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    latencies, raws, parsed, ok = run_conversation(
                        client, args.model, story, thinking_on, args.no_extra_body
                    )
                    if ok:
                        ic = parsed["turn1"]["initial_creativity"]
                        rc = parsed["turn3"]["reflective_creativity"]
                        total_ms = latencies['turn1'] + latencies['turn2'] + latencies['turn3']
                        print(f"IC={ic} RC={rc} ({total_ms}ms)")
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
                        success = True
                        break
                    else:
                        if attempt < MAX_RETRIES:
                            print(f"RETRY {attempt}/{MAX_RETRIES} (bad range or parse)...", end=" ", flush=True)
                        else:
                            print(f"FAILED after {MAX_RETRIES} attempts")
                            for turn in ["turn1", "turn2", "turn3"]:
                                if turn in raws:
                                    print(f"--- RAW {turn.upper()} (START) ---")
                                    print(raws[turn][:500] + "...")
                                    print(f"--- RAW {turn.upper()} (TAIL) ---")
                                    print("..." + raws[turn][-500:])
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        print(f"RETRY {attempt}/{MAX_RETRIES} ({e})...", end=" ", flush=True)
                    else:
                        print(f"ERROR after {MAX_RETRIES} attempts: {e}")

            if not success:
                results.append({
                    "story_id": story["id"],
                    "thinking_mode": thinking_on,
                    "parse_ok": False,
                    "raw_responses": raws if 'raws' in dir() else {}
                })

            with open(OUTPUT_FILE, "w") as f:
                json.dump({"model": args.model, "thinking_mode": thinking_on, "results": results}, f, indent=2)

    print(f"\nDone. Both modes complete for {args.model}.")

if __name__ == "__main__":
    main()
