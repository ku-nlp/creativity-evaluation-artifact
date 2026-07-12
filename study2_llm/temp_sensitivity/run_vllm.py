"""
Temperature Sensitivity — open models on vLLM.

Runs the full 3-turn protocol N times per story (default 3) on a configurable
subset (default: all 80 stories) at the model's normal recommended temperature.
The point is to measure within-rater σ across runs — to show that LLM judges
are effectively deterministic on this task.

Output:  study2_llm/temp_sensitivity/results/temp_sens_<label>.json
         (one record per (story_id, run_idx) — same shape as main results files,
          plus a `run_idx` field)

Usage on the GPU server (open models):
    python study2_llm/temp_sensitivity/run_vllm.py \
        --url http://localhost:8000/v1 \
        --model Qwen/Qwen3.5-4B \
        --label qwen35_4b_nothink \
        --mode nothink \
        --repeats 3

    python study2_llm/temp_sensitivity/run_vllm.py \
        --url http://localhost:8000/v1 \
        --model google/gemma-4-31b-it \
        --label gemma4_31b_nothink \
        --mode nothink \
        --repeats 3
"""
import os, sys, json, argparse, time
from pathlib import Path
from openai import OpenAI

# Reuse the existing infrastructure
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rating_scripts"))
from run_judges import run_conversation, validate_turn1, validate_turn2, validate_turn3  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True, help="HF model id served by vLLM")
    ap.add_argument("--label", required=True, help="Short label for output filename")
    ap.add_argument("--stories", default="study2_llm/data/stories_80.json")
    ap.add_argument("--mode", choices=["nothink", "think"], default="nothink")
    ap.add_argument("--no-extra-body", action="store_true",
                    help="Skip Qwen-specific extra_body params (use for Llama)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Override temperature only (other sampling params stay "
                         "at study defaults). Default None = study temperature. "
                         "Output file gets a _t<temp> suffix.")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-stories", type=int, default=None,
                    help="Cap number of stories (debug); default = all")
    ap.add_argument("--max-retries", type=int, default=3,
                    help="Retry attempts per (story, run_idx) when validation fails")
    args = ap.parse_args()

    REPO_ROOT = HERE.parent.parent
    stories_path = (REPO_ROOT / args.stories) if not Path(args.stories).is_absolute() else Path(args.stories)
    with open(stories_path) as f:
        stories = json.load(f)
    if args.max_stories:
        stories = stories[:args.max_stories]

    OUT_DIR = HERE / "results"
    OUT_DIR.mkdir(exist_ok=True)
    suffix = f"_t{args.temperature:g}" if args.temperature is not None else ""
    OUT_PATH = OUT_DIR / f"temp_sens_{args.label}{suffix}.json"

    client = OpenAI(base_url=args.url, api_key="EMPTY")
    thinking_on = (args.mode == "think")

    print(f"=== Temperature sensitivity ===")
    print(f"  model: {args.model}    label: {args.label}")
    print(f"  temperature: {'study default' if args.temperature is None else args.temperature}")
    print(f"  url: {args.url}    mode: {args.mode}    repeats: {args.repeats}")
    print(f"  stories: {len(stories)}  total calls: {len(stories) * args.repeats}")
    print(f"  output: {OUT_PATH}")
    print()

    # Resume support — keep OK records, drop failed records so they get retried
    results = []
    if OUT_PATH.exists():
        try:
            existing = json.load(open(OUT_PATH))
            all_rows = existing.get("results", [])
            results = [r for r in all_rows if r.get("parse_ok")]
            n_dropped = len(all_rows) - len(results)
            done = {(r["story_id"], r["run_idx"]) for r in results}
            print(f"  RESUME: {len(done)} OK pairs kept; {n_dropped} failed pairs dropped (will retry).\n")
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

            ok = False
            attempts = 0
            latencies, raws, parsed = {}, {}, {}
            err_msg = None
            t0 = time.time()
            while attempts < args.max_retries and not ok:
                attempts += 1
                try:
                    latencies, raws, parsed, ok = run_conversation(
                        client, args.model, story, thinking_on,
                        no_extra_body=args.no_extra_body,
                        temperature=args.temperature,
                    )
                except Exception as e:
                    err_msg = str(e)
                    ok = False
            elapsed = time.time() - t0

            rec = {
                "story_id": sid,
                "tone": story.get("tone"),
                "run_idx": run_idx,
                "temperature": args.temperature,  # None = study default
                "parse_ok": ok,
                "attempts": attempts,
                "latency_ms": latencies,
            }
            if ok:
                rec["scores"] = {
                    "initial_creativity":   parsed["turn1"]["initial_creativity"],
                    "enjoyment":            parsed["turn1"]["enjoyment"],
                    "reflective_creativity":parsed["turn3"]["reflective_creativity"],
                    "sub_components":       {k: parsed["turn2"][k] for k in parsed["turn2"]},
                }
            else:
                rec["partial"] = {k: parsed.get(k) for k in ("turn1", "turn2", "turn3")}
                if err_msg:
                    rec["error"] = err_msg

            results.append(rec)

            marker = "✓" if ok else "✗"
            ic = rec["scores"]["initial_creativity"] if ok else "-"
            rc = rec["scores"]["reflective_creativity"] if ok else "-"
            atag = f"a{attempts}" if attempts > 1 else "  "
            print(f"  [{s_idx+1:>2}/{len(stories)}] {sid[:30]:<30} run={run_idx} {atag}  "
                  f"{marker}  IC={ic}  RC={rc}   ({elapsed:.1f}s)")

            # Incremental save every 10 records
            if len(results) % 10 == 0:
                json.dump({"model": args.model, "label": args.label, "mode": args.mode,
                           "repeats": args.repeats, "n_stories": len(stories),
                           "results": results},
                          open(OUT_PATH, "w"), indent=2)

    # Final save
    json.dump({"model": args.model, "label": args.label, "mode": args.mode,
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
