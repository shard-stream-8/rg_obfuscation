#!/usr/bin/env bash
# Simple helper to run train.py sequentially multiple times.
# Usage:  scripts/run_many.sh <config_file.py> [num_runs]
#   <config_file.py> : Path relative to the ./configs directory (required)
#   [num_runs]       : Number of times to repeat (default: 1)
#
# Example: scripts/run_many.sh acre_shoggoth.yaml 5

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_file.yaml> [num_runs]" >&2
  exit 1
fi

CONFIG_REL="$1"           # e.g. "acre_shoggoth.yaml"
RUNS="${2:-1}"            # default 1 if not provided

CONFIG_PATH="configs/${CONFIG_REL}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Error: Config file '${CONFIG_PATH}' not found." >&2
  exit 1
fi

# Ensure RUNS is a positive integer
if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [[ "$RUNS" -le 0 ]]; then
  echo "Error: num_runs must be a positive integer." >&2
  exit 1
fi

# Auto-detach the script unless already running in detached mode
if [[ -z "${RUN_MANY_DETACHED:-}" ]]; then
  LOG_FILE="run_many_$(date +%Y%m%d_%H%M%S).log"
  echo "Detaching run_many.sh; logs will be written to ${LOG_FILE}"
  nohup env RUN_MANY_DETACHED=1 "$0" "$@" > "${LOG_FILE}" 2>&1 &
  echo "Background PID $!; you can safely disconnect. Tail with: tail -f ${LOG_FILE}"
  exit 0
fi

for ((i=1; i<=RUNS; i++)); do
  echo "=== Starting run ${i}/${RUNS} with config '${CONFIG_REL}' ==="
  # nohup writes to nohup.out in the current directory (appended)
  nohup python train.py "${CONFIG_PATH}"
  echo "Finished run ${i}/${RUNS}. Output appended to nohup.out"
  echo "-------------------------------------------"
  sleep 1  # small delay before next run (optional)
done

echo "All ${RUNS} runs completed." 