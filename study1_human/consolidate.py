"""
Consolidate all conversational 3-turn results into a single DataFrame/CSV.
17 conditions × 12 stories = 204 rows.
"""

import json
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# All result files with metadata
# (path, family, size, reasoning_mode)
RESULT_FILES = [
    # --- Meta (Llama) ---
    ("study1_human/data/raw_json/llama31_8b.json",              "Meta",      "8B",   "none"),
    ("study1_human/data/raw_json/llama31_70b.json",             "Meta",      "70B",  "none"),
    # --- Alibaba (Qwen) ---
    ("study1_human/data/raw_json/qwen3_8b_thinkingoff.json",   "Alibaba",   "8B",   "off"),
    ("study1_human/data/raw_json/qwen3_8b_thinkingon.json",    "Alibaba",   "8B",   "on"),
    ("study1_human/data/raw_json/qwen3_32b_thinkingoff.json",  "Alibaba",   "32B",  "off"),
    ("study1_human/data/raw_json/qwen3_32b_thinkingon.json",   "Alibaba",   "32B",  "on"),
    # --- Microsoft (Phi) ---
    ("study1_human/data/raw_json/phi4_standard.json",           "Microsoft", "14B",  "none"),
    # --- Google (Gemini 2.5 Pro — thinking_budget) ---
    ("study1_human/data/raw_json/gemini_25_pro.json",           "Google",    "2.5pro", "budget_default"),
    ("study1_human/data/raw_json/gemini_25_pro_budget128.json", "Google",    "2.5pro", "budget_128"),
    ("study1_human/data/raw_json/gemini_25_pro_budget32768.json","Google",   "2.5pro", "budget_32768"),
    # --- Google (Gemini 3 Pro Preview — default only) ---
    ("study1_human/data/raw_json/gemini_3_pro_preview.json",    "Google",    "3pro_preview", "default"),
    # --- Google (Gemini 3 Pro Preview — thinking_budget) ---
    ("study1_human/data/raw_json/gemini_3_pro_preview_budget128.json", "Google", "3pro_preview", "budget_128"),
    # --- Google (Gemini 3.1 Pro — thinking_level) ---
    ("study1_human/data/raw_json/gemini_31_pro_high.json",      "Google",    "3.1pro", "level_high"),
    ("study1_human/data/raw_json/gemini_31_pro_medium.json",    "Google",    "3.1pro", "level_medium"),
    ("study1_human/data/raw_json/gemini_31_pro_low.json",       "Google",    "3.1pro", "level_low"),
    # --- OpenAI ---
    ("study1_human/data/raw_json/gpt41.json",                   "OpenAI",    "large", "none"),
    ("study1_human/data/raw_json/o4_mini.json",                 "OpenAI",    "mini",  "reasoning"),
    # --- OpenAI (GPT-5.4 — reasoning_effort) ---
    ("study1_human/data/raw_json/gpt54_none.json",              "OpenAI",    "5.4",   "effort_none"),
    ("study1_human/data/raw_json/gpt54_medium.json",            "OpenAI",    "5.4",   "effort_medium"),
    ("study1_human/data/raw_json/gpt54_high.json",              "OpenAI",    "5.4",   "effort_high"),
]

SUB_COMPONENTS = [
    "emotional_impact", "topic_fidelity", "vocabulary_freshness",
    "plot_uniqueness", "surprise", "empathy", "thought_provocation",
    "engagement", "stylistic_quality", "logical_coherence", "tone_fidelity",
]

rows = []

for rel_path, family, size, reasoning in RESULT_FILES:
    fpath = BASE / rel_path
    if not fpath.exists():
        print(f"WARNING: {fpath} not found, skipping")
        continue

    with open(fpath) as f:
        data = json.load(f)

    label = data.get("label", fpath.stem)
    model_id = data.get("model", "")

    for entry in data["results"]:
        if not entry.get("parse_ok", False):
            continue

        scores = entry["scores"]
        subs = scores.get("sub_components", {})

        # Extract topic and tone from story_id
        story_id = entry["story_id"]
        parts = story_id.rsplit("_", 1)
        tone = parts[-1] if len(parts) == 2 else "unknown"
        topic = parts[0] if len(parts) == 2 else story_id

        row = {
            "model_label": label,
            "model_id": model_id,
            "family": family,
            "size": size,
            "reasoning_mode": reasoning,
            "story_id": story_id,
            "topic": topic,
            "tone": tone,
            "IC": scores.get("initial_creativity"),
            "RC": scores.get("reflective_creativity"),
            "enjoyment": scores.get("enjoyment"),
            "gk_flip": entry.get("gk_flip", 0),
            "flip_abs": abs(entry.get("gk_flip", 0)),
            "flipped": 1 if entry.get("gk_flip", 0) != 0 else 0,
            "sub_mean": None,
        }

        # Add individual sub-components
        for sc in SUB_COMPONENTS:
            row[sc] = subs.get(sc)

        # Compute sub-component mean
        sc_vals = [subs[k] for k in SUB_COMPONENTS if k in subs and subs[k] is not None]
        if sc_vals:
            row["sub_mean"] = round(sum(sc_vals) / len(sc_vals), 2)

        rows.append(row)

df = pd.DataFrame(rows)

# Derived columns
df["flip_direction"] = df["gk_flip"].apply(lambda x: "up" if x > 0 else ("down" if x < 0 else "none"))
df["IC_RC_diff"] = df["RC"] - df["IC"]

# Sort
df = df.sort_values(["family", "model_label", "story_id"]).reset_index(drop=True)

# Save
out_dir = BASE / "study1_human/data"
out_dir.mkdir(exist_ok=True)
out_csv = out_dir / "llm_ratings.csv"
df.to_csv(out_csv, index=False)

print(f"Consolidated {len(df)} rows from {df['model_label'].nunique()} models")
print(f"Saved to: {out_csv}")
print()

# Quick summary
print("=== Model Summary ===")
summary = df.groupby("model_label").agg(
    family=("family", "first"),
    n=("story_id", "count"),
    IC_mean=("IC", "mean"),
    RC_mean=("RC", "mean"),
    flip_rate=("flipped", "mean"),
    mean_flip=("gk_flip", "mean"),
    sub_mean=("sub_mean", "mean"),
).round(2)
print(summary.to_string())

print()
print("=== Tone Summary (pooled) ===")
tone_summary = df.groupby("tone").agg(
    IC_mean=("IC", "mean"),
    RC_mean=("RC", "mean"),
    flip_rate=("flipped", "mean"),
    mean_flip=("gk_flip", "mean"),
).round(2)
print(tone_summary.to_string())

print()
print(f"\nColumns: {list(df.columns)}")
