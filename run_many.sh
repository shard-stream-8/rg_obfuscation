#!/usr/bin/env bash

# run_many.sh
# Usage: bash run_many.sh <CONFIG_PATH> <NUM_RUNS>
# Runs the training script NUM_RUNS times sequentially using nohup so each run
# survives hangups and appends output to nohup.out.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <CONFIG_PATH> <NUM_RUNS>"
  exit 1
fi

CONFIG_FILE=$1
NUM_RUNS=$2

for i in $(seq 1 "$NUM_RUNS"); do
  echo "============================="
  echo "Starting run $i/$NUM_RUNS with config $CONFIG_FILE"
  echo "(Output will append to nohup.out)"
  echo "-----------------------------"

  # Run the training script. The command is placed inside an 'if' so that
  # we can capture non-zero exit codes without the script aborting due to
  # 'set -e'. This lets the loop continue even if one run fails.
  if nohup python train.py "$CONFIG_FILE"; then
    echo "Run $i completed successfully (exit code 0)."
  else
    echo "Run $i FAILED (exit code $?). Continuing to next run."
  fi

  echo "============================="
  echo

done

echo "All $NUM_RUNS runs attempted."