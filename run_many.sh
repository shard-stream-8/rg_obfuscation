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

  # Run in foreground; nohup ensures the process ignores SIGHUP if the terminal closes.
  nohup python train.py "$CONFIG_FILE"

  echo "Run $i completed (exit code $?)."
  echo "============================="
  echo

done

echo "All $NUM_RUNS runs completed." 