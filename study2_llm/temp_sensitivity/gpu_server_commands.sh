#!/bin/bash
# Open-model repeated-sampling + temperature sweep on the GPU server (reviewer-requested sampling robustness check).
#
# Design (mirrors the proprietary runs done via API on 2026-07-09):
#   - 10 repeats at the study-default temperature (extends the existing
#     3-repeat temp_sens files; resume support picks up runs 0-2 for free)
#   - temperature sweep {0.0, 0.5, 1.0, 1.5} x 3 repeats, temperature only,
#     all other sampling params held at study values
#   - same 3 models as the existing files: Qwen3.5-4B NT (noisiest, SD 0.53),
#     Gemma4-26B NT (most deterministic, SD 0.04), Llama3.1-70B NT (middle)
#
# Serve each model with vLLM first, then run its block. Not a driver script;
# copy-paste one block at a time (each model needs its own vLLM server).
set -u
cd "$(dirname "$0")/../.."
RUN="python study2_llm/temp_sensitivity/run_vllm.py --url http://localhost:8000/v1"

# ---------- Qwen3.5-4B (nothink; study temp 0.7) ----------
# vllm serve Qwen/Qwen3.5-4B
$RUN --model Qwen/Qwen3.5-4B --label qwen35_4b_nothink --mode nothink --repeats 10
for T in 0.0 0.5 1.0 1.5; do
  $RUN --model Qwen/Qwen3.5-4B --label qwen35_4b_nothink --mode nothink \
       --temperature $T --repeats 3
done

# ---------- Gemma4-26B (nothink; study temp 0.7) ----------
# vllm serve google/gemma-4-26b-a4b-it
$RUN --model google/gemma-4-26b-a4b-it --label gemma4_26b_a4b_nothink --mode nothink --repeats 10
for T in 0.0 0.5 1.0 1.5; do
  $RUN --model google/gemma-4-26b-a4b-it --label gemma4_26b_a4b_nothink --mode nothink \
       --temperature $T --repeats 3
done

# ---------- Llama3.1-70B (no extra body; study temp 0.7) ----------
# vllm serve meta-llama/Llama-3.1-70B-Instruct
$RUN --model meta-llama/Llama-3.1-70B-Instruct --label llama31_70b_nothink \
     --mode nothink --no-extra-body --repeats 10
for T in 0.0 0.5 1.0 1.5; do
  $RUN --model meta-llama/Llama-3.1-70B-Instruct --label llama31_70b_nothink \
       --mode nothink --no-extra-body --temperature $T --repeats 3
done

# When done, rsync study2_llm/temp_sensitivity/results/ back and run:
#   python study2_llm/temp_sensitivity/analyze_sampling.py
