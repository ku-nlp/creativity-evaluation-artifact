"""
Prompt Sensitivity — open models on vLLM.

Mirrors the original Gemini/GPT-4.1-mini sensitivity study (4 variants × 12 stories)
but extended to open models served by vLLM. Tests whether the lack of revision
behaviour observed in the main study generalises across Turn-3 prompt wording.

Variants (matching the original sensitivity study):
  v1 — neutral, both options explicit ("you may revise or keep — both equally valid")
  v2 — minimal, no mention of revision ("based on everything you considered")
  v4 — original baseline ("has your view changed?")
(v3 in the original study was identical to v1 in code, so we drop it.)

12 stories: first 12 of stories_80.json — the original sensitivity story texts
            (3 topics × 4 tones: ai_shutdown_*, midnight_store_*, the_heist_*).

Output:  study2_llm/prompt_sensitivity/results/sensitivity_<label>.json
         (one record per (story_id, variant), shape matching
          sensitivity_gemini_2.5_pro.json so the existing analyzer can read it.)

Usage on the GPU server:
    python study2_llm/prompt_sensitivity/run_open.py \
        --url http://localhost:8001/v1 \
        --model Qwen/Qwen3.5-4B \
        --label qwen35_4b_nothink \
        --mode nothink

    python study2_llm/prompt_sensitivity/run_open.py \
        --url http://localhost:8000/v1 \
        --model google/gemma-4-26b-a4b-it \
        --label gemma4_26b_a4b_nothink \
        --mode nothink --no-extra-body
"""
import os, sys, json, argparse, time
from pathlib import Path
from openai import OpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rating_scripts"))
from run_judges import (
    _make_request, parse_json, fix_keys,
    TURN1_KEYS, TURN2_KEYS,
    validate_turn1, validate_turn2, validate_turn3,
)
from shared_prompts import (
    SYSTEM_PROMPT, build_turn1_prompt, build_turn2_prompt, build_turn3_prompt,
)

# 3 distinct Turn-3 variants (v3 was a duplicate of v1 in the original code)
TURN3_VARIANTS = {
    "v1": """Thank you. Having reflected on the story across all those dimensions, please give your final overall creativity rating.

"Overall, I consider this story to be creative." (1–7)

You may revise your initial score or keep it — both are equally valid. Please briefly explain your reasoning.

Respond ONLY with a JSON object:
{"reflective_creativity": <int>, "reasoning": "<string>"}""",

    "v2": """Thank you. Based on everything you have considered, please give your final overall creativity rating.

"Overall, I consider this story to be creative." (1–7)

Please briefly explain your final score.

Respond ONLY with a JSON object:
{"reflective_creativity": <int>, "reasoning": "<string>"}""",

    "v4": """Thank you. Having now considered the story in detail across all those dimensions, I'd like you to give one final overall rating:

"Overall, I consider this story to be creative." (1-7)

Has your view changed from your initial rating? Provide your final score and briefly explain.

Respond ONLY with a JSON object:
{"reflective_creativity": <int>, "reasoning": "<string>"}""",
}


def run_one_variant(client, model_name, story, variant, thinking_on, no_extra_body, max_retries=3):
    """Run T1, T2, then T3 with the specified variant. Returns (record_dict, ok)."""
    err_msg = None
    for attempt in range(1, max_retries + 1):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Turn 1
        try:
            messages.append({"role": "user", "content": build_turn1_prompt(story)})
            resp1 = _make_request(client, model_name, messages, thinking_on, no_extra_body)
            raw1 = resp1.choices[0].message.content or ""
            t1 = fix_keys(parse_json(raw1), TURN1_KEYS)
            if not validate_turn1(t1):
                continue
            messages.append({"role": "assistant", "content": json.dumps(t1)})

            # Turn 2
            messages.append({"role": "user", "content": build_turn2_prompt()})
            resp2 = _make_request(client, model_name, messages, thinking_on, no_extra_body)
            raw2 = resp2.choices[0].message.content or ""
            t2 = fix_keys(parse_json(raw2), TURN2_KEYS)
            if not validate_turn2(t2):
                continue
            messages.append({"role": "assistant", "content": json.dumps(t2)})

            # Turn 3 — variant
            messages.append({"role": "user", "content": TURN3_VARIANTS[variant]})
            resp3 = _make_request(client, model_name, messages, thinking_on, no_extra_body)
            raw3 = resp3.choices[0].message.content or ""
            t3 = parse_json(raw3)
            if not validate_turn3(t3):
                continue

            ic = t1["initial_creativity"]
            rc = t3["reflective_creativity"]
            return {
                "story_id": story["id"],
                "topic": story.get("topic"),
                "tone": story.get("tone"),
                "variant": variant,
                "IC": ic,
                "enjoyment": t1["enjoyment"],
                "RC": rc,
                "reasoning": t3.get("reasoning", ""),
                "flipped": ic != rc,
                "IC_RC_diff": rc - ic,
                "sub": t2,
                "attempts": attempt,
                "parse_ok": True,
            }, True
        except Exception as e:
            err_msg = str(e)
            continue
    return {
        "story_id": story["id"],
        "topic": story.get("topic"),
        "tone": story.get("tone"),
        "variant": variant,
        "parse_ok": False,
        "attempts": max_retries,
        "error": err_msg,
    }, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, help="Short label for output filename")
    ap.add_argument("--stories", default="study2_llm/data/stories_80.json")
    ap.add_argument("--mode", choices=["nothink", "think"], default="nothink")
    ap.add_argument("--no-extra-body", action="store_true",
                    help="Skip Qwen-specific extra_body params (use for Gemma, Llama)")
    ap.add_argument("--variants", default="v1,v2,v4",
                    help="Comma-separated variants to run (default: v1,v2,v4)")
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()

    REPO_ROOT = HERE.parent.parent
    stories_path = (REPO_ROOT / args.stories) if not Path(args.stories).is_absolute() else Path(args.stories)
    all_stories = json.load(open(stories_path))

    # Pick the 12 stories matching the original sensitivity set:
    # 3 topics × 4 tones using ai_shutdown_*, midnight_store_*, the_heist_*
    target_prefixes = ("ai_shutdown_", "midnight_store_", "the_heist_")
    stories = [s for s in all_stories if any(s["id"].startswith(p) for p in target_prefixes)]
    if len(stories) != 12:
        print(f"  WARN: expected 12 sensitivity stories, found {len(stories)}")

    OUT_DIR = HERE / "results"
    OUT_DIR.mkdir(exist_ok=True)
    OUT_PATH = OUT_DIR / f"sensitivity_{args.label}.json"

    variants = [v.strip() for v in args.variants.split(",")]
    for v in variants:
        if v not in TURN3_VARIANTS:
            raise SystemExit(f"Unknown variant '{v}'. Available: {list(TURN3_VARIANTS)}")

    client = OpenAI(base_url=args.url, api_key="EMPTY")
    thinking_on = (args.mode == "think")

    print(f"=== Prompt sensitivity (open) ===")
    print(f"  model: {args.model}    label: {args.label}")
    print(f"  url: {args.url}    mode: {args.mode}")
    print(f"  variants: {variants}    stories: {len(stories)}")
    print(f"  total calls: {len(stories) * len(variants)}")
    print(f"  output: {OUT_PATH}\n")

    # Resume — keep OK records, drop failed ones
    results = []
    if OUT_PATH.exists():
        try:
            rows = json.load(open(OUT_PATH))
            if isinstance(rows, dict) and "results" in rows:
                rows = rows["results"]
            results = [r for r in rows if r.get("parse_ok")]
            dropped = len(rows) - len(results)
            done = {(r["story_id"], r["variant"]) for r in results}
            print(f"  RESUME: {len(done)} OK kept; {dropped} failed dropped.\n")
        except Exception as e:
            print(f"  could not parse existing file ({e}); fresh start.\n")
            results = []
            done = set()
    else:
        done = set()

    t_start = time.time()
    for v in variants:
        print(f"\n── Variant {v} ─────────────────────────────")
        for s_idx, story in enumerate(stories):
            if (story["id"], v) in done:
                continue
            t0 = time.time()
            rec, ok = run_one_variant(
                client, args.model, story, v,
                thinking_on=thinking_on, no_extra_body=args.no_extra_body,
                max_retries=args.max_retries,
            )
            elapsed = time.time() - t0
            results.append(rec)

            marker = "✓" if ok else "✗"
            ic = rec.get("IC", "-")
            rc = rec.get("RC", "-")
            flip = "↕" if rec.get("flipped") else "="
            atag = f"a{rec.get('attempts',1)}" if rec.get("attempts",1) > 1 else "  "
            print(f"  [{s_idx+1:>2}/{len(stories)}] {story['id'][:30]:<30} "
                  f"{atag}  {marker}  IC={ic}  RC={rc} {flip}   ({elapsed:.1f}s)")

            # Incremental save
            json.dump({"model": args.model, "label": args.label, "mode": args.mode,
                       "variants": variants, "n_stories": len(stories),
                       "results": results},
                      open(OUT_PATH, "w"), indent=2)

    # Final save
    json.dump({"model": args.model, "label": args.label, "mode": args.mode,
               "variants": variants, "n_stories": len(stories),
               "results": results},
              open(OUT_PATH, "w"), indent=2)

    total = time.time() - t_start
    n_ok = sum(1 for r in results if r.get("parse_ok"))
    print(f"\n=== Done in {total/60:.1f} min ===")
    print(f"  {n_ok}/{len(results)} successful parses")
    print(f"  saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
