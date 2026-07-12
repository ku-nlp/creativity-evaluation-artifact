"""
Temperature Sensitivity — Gemini 3 Pro (Low effort).

Runs the full 3-turn protocol N times per story (default 3) on all 80 stories
at the same configuration used in the main study (thinking_level=low, budget=4096,
temperature=1.0). Goal: measure within-rater σ across runs.

Output: study2_llm/temp_sensitivity/results/temp_sens_gemini_3_pro_low.json

Usage:
    source .env  # to load GEMINI_API_KEY
    python study2_llm/temp_sensitivity/run_gemini.py \
        --model gemini-3-pro-preview \
        --label gemini_3_pro_low \
        --thinking-level low \
        --repeats 3
"""
import os, sys, json, argparse, time
from pathlib import Path

# Reuse the existing run_gemini conversation function
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rating_scripts"))
from run_gemini import run_conversation  # noqa
from google import genai


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3-pro-preview")
    ap.add_argument("--label", default="gemini_3_pro_low")
    ap.add_argument("--thinking-level", choices=["none","low","medium","high"], default="low")
    ap.add_argument("--stories", default="study2_llm/data/stories_80.json")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-stories", type=int, default=None)
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set. Run `source .env` first.")
        sys.exit(1)

    REPO_ROOT = HERE.parent.parent
    stories_path = (REPO_ROOT / args.stories) if not Path(args.stories).is_absolute() else Path(args.stories)
    with open(stories_path) as f:
        stories = json.load(f)
    if args.max_stories:
        stories = stories[:args.max_stories]

    OUT_DIR = HERE / "results"
    OUT_DIR.mkdir(exist_ok=True)
    OUT_PATH = OUT_DIR / f"temp_sens_{args.label}.json"

    client = genai.Client(api_key=api_key)

    print(f"=== Gemini temperature sensitivity ===")
    print(f"  model: {args.model}   level: {args.thinking_level}")
    print(f"  stories: {len(stories)}  repeats: {args.repeats}  total calls: {len(stories)*args.repeats}")
    print(f"  output: {OUT_PATH}\n")

    # Resume support
    results = []
    if OUT_PATH.exists():
        try:
            existing = json.load(open(OUT_PATH))
            results = existing.get("results", [])
            done = {(r["story_id"], r["run_idx"]) for r in results if r.get("parse_ok")}
            print(f"  RESUME: {len(done)} (story, run) pairs already done; skipping.\n")
        except Exception as e:
            print(f"  could not parse existing file ({e}); starting fresh.\n")
            results = []
            done = set()
    else:
        done = set()

    t_start = time.time()
    for s_idx, story in enumerate(stories):
        sid = story["id"]
        for run_idx in range(args.repeats):
            if (sid, run_idx) in done:
                continue
            t0 = time.time()
            try:
                latencies, raws, parsed, ok = run_conversation(
                    client, args.model, story, args.thinking_level,
                )
            except Exception as e:
                print(f"  [ERR] {sid} run={run_idx}: {e}")
                results.append({
                    "story_id": sid, "tone": story.get("tone"), "run_idx": run_idx,
                    "thinking_level": args.thinking_level,
                    "parse_ok": False, "error": str(e),
                })
                # Brief backoff on errors (rate limits etc.)
                time.sleep(2)
                continue

            rec = {
                "story_id": sid,
                "tone": story.get("tone"),
                "run_idx": run_idx,
                "thinking_level": args.thinking_level,
                "parse_ok": ok,
                "latency_ms": latencies,
            }
            if ok:
                rec["scores"] = {
                    "initial_creativity":    parsed["turn1"]["initial_creativity"],
                    "enjoyment":             parsed["turn1"]["enjoyment"],
                    "reflective_creativity": parsed["turn3"]["reflective_creativity"],
                    "sub_components":        {k: parsed["turn2"][k] for k in parsed["turn2"]},
                }
            results.append(rec)

            elapsed = time.time() - t0
            marker = "✓" if ok else "✗"
            ic = rec["scores"]["initial_creativity"] if ok else "-"
            rc = rec["scores"]["reflective_creativity"] if ok else "-"
            print(f"  [{s_idx+1:>2}/{len(stories)}] {sid[:30]:<30} run={run_idx}  "
                  f"{marker}  IC={ic}  RC={rc}   ({elapsed:.1f}s)")

            # Incremental save every 5 records (Gemini API can hiccup)
            if len(results) % 5 == 0:
                json.dump({"model": args.model, "label": args.label,
                           "thinking_level": args.thinking_level,
                           "repeats": args.repeats, "n_stories": len(stories),
                           "results": results},
                          open(OUT_PATH, "w"), indent=2)

    # Final save
    json.dump({"model": args.model, "label": args.label,
               "thinking_level": args.thinking_level,
               "repeats": args.repeats, "n_stories": len(stories),
               "results": results},
              open(OUT_PATH, "w"), indent=2)

    total = time.time() - t_start
    n_ok = sum(1 for r in results if r.get("parse_ok"))
    print(f"\n=== Done in {total/60:.1f} min ===")
    print(f"  {n_ok}/{len(results)} successful parses")
    print(f"  saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
