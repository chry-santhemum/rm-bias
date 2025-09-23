#!/usr/bin/env bash
set -u
set -o pipefail

# launch.sh
# A general launcher to run one or more commands sequentially or in parallel,
# with robust error handling, optional per-GPU scheduling, and a final summary.

usage() {
  cat <<'EOF'
Usage:
  launch.sh [options] -- "cmd 1" "cmd 2" ...
  launch.sh [options] -f commands.txt

Options:
  -f FILE                Read commands from FILE (one command per line; # and blank lines ignored)
  -p, --parallel         Run commands in parallel (default: sequential unless -c>1 or --per-gpu)
  -c, --concurrency N    Max concurrent jobs when running in parallel (default: number of GPUs if --per-gpu, else 1)
  -G, --per-gpu          Schedule jobs one per GPU (uses nvidia-smi or CUDA_VISIBLE_DEVICES)
      --second-delay S   Delay S seconds before launching the 2nd job (default 0)
      --step-delay S     Delay S seconds before launching each job after the 2nd (default 0)
  -h, --help             Show this help

Examples:
  launch.sh -- "python a.py" "python b.py --x" "python c.py"
  launch.sh -p -c 4 -f cmds.txt
  launch.sh --per-gpu -f cmds.txt
EOF
}

SECOND_DELAY=0
STEP_DELAY=0
PARALLEL=0
CONCURRENCY=0
PER_GPU=0
CMDS_FILE=""

# Parse args
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f)
      CMDS_FILE=${2:-}; shift 2 ;;
    --second-delay)
      SECOND_DELAY=${2:-0}; shift 2 ;;
    --step-delay)
      STEP_DELAY=${2:-0}; shift 2 ;;
    -p|--parallel)
      PARALLEL=1; shift ;;
    -c|--concurrency)
      CONCURRENCY=${2:-0}; shift 2 ;;
    -G|--per-gpu)
      PER_GPU=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift; while [[ $# -gt 0 ]]; do ARGS+=("$1"); shift; done ;;
    *)
      ARGS+=("$1"); shift ;;
  esac
done

# Collect commands
declare -a COMMANDS
if [[ -n "$CMDS_FILE" ]]; then
  if [[ ! -f "$CMDS_FILE" ]]; then
    echo "Commands file not found: $CMDS_FILE" >&2
    exit 2
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    # Trim
    cmd="${line%%\r}"
    # Skip blanks and comments
    [[ -z "$cmd" || "$cmd" =~ ^[[:space:]]*# ]] && continue
    COMMANDS+=("$cmd")
  done < "$CMDS_FILE"
fi

# Positional commands (after -- or plain args)
if [[ ${#ARGS[@]} -gt 0 ]]; then
  for c in "${ARGS[@]}"; do
    COMMANDS+=("$c")
  done
fi

if [[ ${#COMMANDS[@]} -eq 0 ]]; then
  echo "No commands provided." >&2
  usage
  exit 2
fi

# GPU discovery (optional)
discover_gpus() {
  local -n _out=$1
  _out=()
  if command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t _out < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
    if [[ ${#_out[@]} -eq 0 ]]; then
      if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a _out <<< "${CUDA_VISIBLE_DEVICES}"
      else
        _out=(0)
      fi
    fi
  else
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      IFS=',' read -r -a _out <<< "${CUDA_VISIBLE_DEVICES}"
    else
      _out=(0)
    fi
  fi
}

declare -a GPUS
if [[ $PER_GPU -eq 1 ]]; then
  discover_gpus GPUS
  if [[ ${#GPUS[@]} -eq 0 ]]; then
    GPUS=(0)
  fi
fi

# Decide concurrency
if [[ $PER_GPU -eq 1 ]]; then
  if [[ $CONCURRENCY -le 0 ]]; then
    CONCURRENCY=${#GPUS[@]}
  fi
else
  if [[ $CONCURRENCY -le 0 ]]; then
    CONCURRENCY=$([[ $PARALLEL -eq 1 ]] && echo 9999 || echo 1)
  fi
fi

declare -a pids
declare -a slots_gpu
declare -a statuses

cleanup() {
  trap - INT TERM
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}

on_int() {
  echo "Caught interrupt. Stopping all runs..." >&2
  cleanup
  exit 130
}
trap on_int INT TERM

launch_job() {
  local idx="$1"; shift
  local gpu="$1"; shift
  local cmd="$1"; shift

  if [[ -n "$gpu" ]]; then
    (CUDA_VISIBLE_DEVICES="$gpu" bash -lc "$cmd") &
  else
    (bash -lc "$cmd") &
  fi
  local pid=$!
  pids[$idx]="$pid"
}

# Scheduler
total=${#COMMANDS[@]}
running=0
next=0
success=0
failed=0

assign_gpu_for_slot() {
  local slot="$1"
  if [[ $PER_GPU -eq 0 ]]; then
    echo ""
    return 0
  fi
  local gidx=$(( slot % ${#GPUS[@]} ))
  echo "${GPUS[$gidx]}"
}

maybe_delay() {
  local launched="$1"
  if [[ "$launched" -eq 1 && "$SECOND_DELAY" -gt 0 ]]; then
    sleep "$SECOND_DELAY"
  elif [[ "$launched" -gt 1 && "$STEP_DELAY" -gt 0 ]]; then
    sleep "$STEP_DELAY"
  fi
}

launched=0

while :; do
  # Launch up to CONCURRENCY
  while [[ $running -lt $CONCURRENCY && $next -lt $total ]]; do
    slot=$running
    gpu="$(assign_gpu_for_slot "$slot")"
    maybe_delay "$launched"
    launch_job "$next" "$gpu" "${COMMANDS[$next]}"
    slots_gpu[$next]="$gpu"
    running=$((running + 1))
    launched=$((launched + 1))
    next=$((next + 1))
  done

  # If nothing running and all launched, break
  if [[ $running -eq 0 && $next -ge $total ]]; then
    break
  fi

  # Wait for any job to finish
  wait -n

  # Reap finished jobs
  for i in $(seq 0 $((total-1))); do
    pid_i="${pids[$i]:-}"
    [[ -z "$pid_i" ]] && continue
    if ! kill -0 "$pid_i" 2>/dev/null; then
      # The process has finished, let's get its exit code
      if wait "$pid_i"; then
        statuses[$i]=0
        success=$((success + 1))
      else
        statuses[$i]=1
        failed=$((failed + 1))
      fi
      
      pids[$i]=""
      running=$((running - 1))
    fi
  done
done

trap - INT TERM

echo
echo "Summary:"
echo "  Total:    $total"
echo "  Success:  $success"
echo "  Failed:   $failed"

if [[ $failed -gt 0 ]]; then
  echo
  echo "Failed jobs:"
  for i in $(seq 0 $((total-1))); do
    if [[ "${statuses[$i]:-0}" -ne 0 ]]; then
      echo "--- Job #$i (Failed) ---"
      echo "  CMD: ${COMMANDS[$i]}"
      echo "  GPU: ${slots_gpu[$i]:-N/A}"
    fi
  done
fi

if [[ $failed -gt 0 ]]; then
  exit 1
else
  exit 0
fi