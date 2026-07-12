"""
Repeated-sampling and temperature-sensitivity runs for the proprietary judges
(reviewer-requested "larger repeated-sampling design").

API facts (probed 2026-07-09):
  - gpt-5.5 with reasoning_effort=high (the main-study config) REJECTS any
    non-default temperature: 400 "Only the default (1) value is supported".
    So GPT-5.5 gets repeated sampling only, at the study config.
  - gemini-3-pro-preview is RETIRED (404). The paper's Gemini 3 conditions
    cannot be re-queried; we use gemini-3.1-pro-preview (also a main-study
    condition, thinking low) for both repeats and the temperature sweep.
  - gemini-3.1-pro-preview accepts temperature in [0.0, 2.0]; API default 1.0
    (the main-study value).

Planned invocations (see run_all_proprietary.sh):
  gemini t=1.0 x10 repeats; gemini t in {0.0, 0.5, 1.5, 2.0} x3 repeats;
  gpt (default temp) x10 repeats. All on the full 80-story set.

Output: results/sampling_<label>.json, one record per (story_id, run_idx),
same shape as the main results files plus run_idx and temperature fields.
Resume-safe: OK records are kept, failed ones retried on rerun.
"""
import os, sys, json, argparse, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rating_scripts"))
from shared_prompts import (SYSTEM_PROMPT, build_turn1_prompt,
                            build_turn2_prompt, build_turn3_prompt)  # noqa
from run_openai import (parse_json, validate_turn1, validate_turn2,
                        validate_turn3)  # noqa

GEMINI_MODEL = "gemini-3.1-pro-preview"
GPT_MODEL = "gpt-5.5"
GEMINI_THINKING_BUDGET = 4096  # main-study "low" config


def gemini_conversation(client, story, temperature):
    from google.genai import types
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=temperature,
        max_output_tokens=16384,
        thinking_config=types.ThinkingConfig(
            thinking_budget=GEMINI_THINKING_BUDGET, include_thoughts=True),
    )
    chat = client.chats.create(model=GEMINI_MODEL, config=cfg)
    parsed = {}
    for turn, (prompt, validate) in enumerate([
            (build_turn1_prompt(story), validate_turn1),
            (build_turn2_prompt(), validate_turn2),
            (build_turn3_prompt(), validate_turn3)], start=1):
        resp = chat.send_message(prompt)
        parsed[f"turn{turn}"] = parse_json(resp.text or "")
        if not validate(parsed[f"turn{turn}"]):
            return parsed, False
    return parsed, True


def gpt_conversation(client, story, temperature):
    # temperature is ignored: locked to default at reasoning_effort=high
    kwargs = {"model": GPT_MODEL, "reasoning_effort": "high"}
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    parsed = {}
    for turn, (prompt, validate) in enumerate([
            (build_turn1_prompt(story), validate_turn1),
            (build_turn2_prompt(), validate_turn2),
            (build_turn3_prompt(), validate_turn3)], start=1):
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(**kwargs, messages=messages)
        raw = resp.choices[0].message.content or ""
        parsed[f"turn{turn}"] = parse_json(raw)
        if not validate(parsed[f"turn{turn}"]):
            return parsed, False
        messages.append({"role": "assistant",
                         "content": json.dumps(parsed[f"turn{turn}"])})
    return parsed, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["gemini", "gpt"], required=True)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="gemini only; gpt is locked to the API default")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--label", default=None)
    ap.add_argument("--stories", default="study2_llm/data/stories_80.json")
    ap.add_argument("--max-stories", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()

    if args.provider == "gemini":
        from google import genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("GEMINI_API_KEY not set. `source .env` first.")
        client = genai.Client(api_key=key)
        convo = gemini_conversation
        model = GEMINI_MODEL
        default_label = f"gemini31_low_t{args.temperature:g}"
    else:
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY not set. `source .env` first.")
        client = OpenAI()
        convo = gpt_conversation
        model = GPT_MODEL
        default_label = "gpt55_high_default"

    label = args.label or default_label
    REPO_ROOT = HERE.parent.parent
    stories_path = Path(args.stories)
    if not stories_path.is_absolute():
        stories_path = REPO_ROOT / stories_path
    stories = json.load(open(stories_path))
    if args.max_stories:
        stories = stories[:args.max_stories]

    OUT_DIR = HERE / "results"
    OUT_DIR.mkdir(exist_ok=True)
    OUT_PATH = OUT_DIR / f"sampling_{label}.json"

    results, done = [], set()
    if OUT_PATH.exists():
        try:
            old = json.load(open(OUT_PATH)).get("results", [])
            results = [r for r in old if r.get("parse_ok")]
            done = {(r["story_id"], r["run_idx"]) for r in results}
            print(f"RESUME: {len(done)} OK pairs kept, "
                  f"{len(old) - len(results)} failed pairs will be retried.")
        except Exception as e:
            print(f"could not parse existing file ({e}); starting fresh")

    jobs = [(s, i) for s in stories for i in range(args.repeats)
            if (s["id"], i) not in done]
    print(f"model={model} temp={args.temperature} repeats={args.repeats} "
          f"workers={args.workers}")
    print(f"jobs remaining: {len(jobs)} conversations -> {OUT_PATH}")

    lock = threading.Lock()
    meta = {"model": model, "label": label, "provider": args.provider,
            "temperature": args.temperature, "repeats": args.repeats,
            "n_stories": len(stories)}

    def save():
        json.dump({**meta, "results": results}, open(OUT_PATH, "w"), indent=2)

    def work(job):
        story, run_idx = job
        parsed, ok, err = {}, False, None
        for attempt in range(args.max_retries):
            try:
                parsed, ok = convo(client, story, args.temperature)
                if ok:
                    break
            except Exception as e:
                err = str(e)
                time.sleep(2 * (attempt + 1))  # backoff on 429/5xx
        rec = {"story_id": story["id"], "tone": story.get("tone"),
               "run_idx": run_idx, "temperature": args.temperature,
               "parse_ok": ok}
        if ok:
            rec["scores"] = {
                "initial_creativity": parsed["turn1"]["initial_creativity"],
                "enjoyment": parsed["turn1"]["enjoyment"],
                "reflective_creativity": parsed["turn3"]["reflective_creativity"],
                "sub_components": dict(parsed["turn2"]),
            }
        elif err:
            rec["error"] = err
        return rec

    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, j): j for j in jobs}
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                results.append(rec)
                n_done += 1
                if n_done % 10 == 0:
                    save()
                    rate = n_done / (time.time() - t0)
                    eta = (len(jobs) - n_done) / rate / 60 if rate else 0
                    print(f"  {n_done}/{len(jobs)} done "
                          f"({100*sum(r['parse_ok'] for r in results)/len(results):.0f}% ok, "
                          f"ETA {eta:.0f} min)", flush=True)
    save()
    n_ok = sum(r["parse_ok"] for r in results)
    print(f"DONE in {(time.time()-t0)/60:.1f} min: {n_ok}/{len(results)} OK "
          f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
