#!/usr/bin/env bash
set -u
set -o pipefail

# Define your sequential runs here. Each entry is a full set of args.
# Add/remove lines as needed.
RUNS=(
  "--breadth 20 --num_steps 6"
  "--breadth 15 --num_steps 8"
  "--breadth 10 --num_steps 12"
  "--breadth 6 --num_steps 20"
)

mkdir -p logs/bon_iter

total=${#RUNS[@]}
success=0
failed=0

for i in "${!RUNS[@]}"; do
  idx=$((i+1))
  args=${RUNS[$i]}
  ts=$(date +%Y%m%d-%H%M%S)
  desc=$(echo ${args} | awk '{print "b"$2"-n"$4}')
  log="logs/bon_iter/${ts}-${desc}.log"

  echo "[${ts}] Starting run ${idx}/${total}: python bon_iter.py ${args}"
  # Stream to console and save to log, capture python's exit code via PIPESTATUS
  python bon_iter.py ${args} 2>&1 | tee "${log}"
  code=${PIPESTATUS[0]}

  if [ ${code} -ne 0 ]; then
    echo "Run ${idx} FAILED (exit ${code}). Log: ${log}"
    failed=$((failed+1))
  else
    echo "Run ${idx} succeeded. Log: ${log}"
    success=$((success+1))
  fi
done

echo "All runs finished. Success: ${success}, Failed: ${failed}, Total: ${total}"
