#!/usr/bin/env bash
set -uo pipefail

# Command to run per GPU
cmd=(python bon_iter.py --breadth 6 --num_steps 1)

# Optional staggering between launches
# SECOND_DELAY: delay (s) before launching the 2nd run (default 0)
# STEP_DELAY: delay (s) before launching every run after the 2nd (default 0)
SECOND_DELAY="${SECOND_DELAY:-0}"
STEP_DELAY="${STEP_DELAY:-0}"

# Discover GPUs
declare -a gpus
if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
  if [ ${#gpus[@]} -eq 0 ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
      IFS=',' read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES}"
    else
      gpus=(0)
    fi
  fi
else
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES}"
  else
    gpus=(0)
  fi
fi

tmpdir="$(mktemp -d -t flat_baseline.XXXXXX)"
declare -a pids
declare -a gpu_for_pid
declare -a success_files

cleanup() {
  trap - INT TERM
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  rm -rf "$tmpdir"
}

on_int() {
  echo "Caught interrupt. Stopping all runs..."
  cleanup
  exit 130
}
trap on_int INT TERM

# Launch one process per GPU (optionally staggered)
for i in "${!gpus[@]}"; do
  gpu="${gpus[$i]}"
  if [ "$i" -eq 1 ] && [ "$SECOND_DELAY" -gt 0 ]; then
    sleep "$SECOND_DELAY"
  elif [ "$i" -gt 1 ] && [ "$STEP_DELAY" -gt 0 ]; then
    sleep "$STEP_DELAY"
  fi
  success_file="$tmpdir/success_${gpu}.status"
  success_files+=("$success_file")
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$tmpdir/run_${gpu}.out" 2>&1 &
  pid=$!
  pids+=("$pid")
  gpu_for_pid+=("$gpu")
done

# Wait for all and record statuses (don't stop others if one fails)
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  gpu="${gpu_for_pid[$i]}"
  success_file="$tmpdir/success_${gpu}.status"
  if wait "$pid"; then
    echo 0 > "$success_file"
  else
    echo 1 > "$success_file"
  fi
done

# Summarize results
successes=0
for sf in "${success_files[@]}"; do
  if [ -f "$sf" ] && [ "$(cat "$sf")" = "0" ]; then
    successes=$((successes + 1))
  fi
done
total="${#gpus[@]}"
echo "Succeeded: ${successes}/${total}"

# Cleanup temp and set exit code (non-zero if any failed)
cleanup
if [ "$successes" -eq "$total" ]; then
  exit 0
else
  exit 1
fi


