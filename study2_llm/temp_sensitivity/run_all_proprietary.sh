#!/bin/bash
# Full proprietary sampling/temperature-sensitivity plan (reviewer-requested sampling robustness check).
# Usage: source .env && bash study2_llm/temp_sensitivity/run_all_proprietary.sh {gemini|gpt}
# Resume-safe: rerun the same command to retry failures.
set -u
cd "$(dirname "$0")/../.."
PY=venv/bin/python
RUN="$PY study2_llm/temp_sensitivity/run_proprietary_sensitivity.py"

if [ "${1:-}" = "gemini" ]; then
    # 10 repeats at the main-study config (temp 1.0, thinking low)
    $RUN --provider gemini --temperature 1.0 --repeats 10
    # temperature sweep, 3 repeats each (full accepted range 0.0-2.0)
    $RUN --provider gemini --temperature 0.0 --repeats 3
    $RUN --provider gemini --temperature 0.5 --repeats 3
    $RUN --provider gemini --temperature 1.5 --repeats 3
    $RUN --provider gemini --temperature 2.0 --repeats 3
elif [ "${1:-}" = "gpt" ]; then
    # temperature is API-locked at reasoning_effort=high; repeats only
    $RUN --provider gpt --repeats 10
else
    echo "usage: $0 {gemini|gpt}"; exit 1
fi
