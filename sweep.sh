#!/usr/bin/env bash
set -u
set -o pipefail

# Define your sequential runs here. Each entry is a full set of args.
# Add/remove lines as needed.
# Controlling for around 60 system prompts
COMMANDS=(
  # "python bon_iter.py --breadth 20 --num_steps 3"
  # "python bon_iter.py --breadth 10 --num_steps 6"
  # "python bon_iter.py --breadth 6 --num_steps 10"
  # "python bon_iter.py --breadth 3 --num_steps 20"
  "python evo_new_prompts.py --N_pop 20 --M_var 3 --num_steps 1"
  "python evo_new_prompts.py --N_pop 10 --M_var 6 --num_steps 1"
  "python evo_new_prompts.py --N_pop 10 --M_var 2 --num_steps 3"
  "python evo_new_prompts.py --N_pop 4 --M_var 5 --num_steps 3"
  "python evo_new_prompts.py --N_pop 3 --M_var 2 --num_steps 10"
  "python evo_new_prompts.py --N_pop 8 --M_var 4 --num_steps 10"
)

total=${#COMMANDS[@]}
success=0
failed=0

for i in "${!COMMANDS[@]}"; do
  cmd="${COMMANDS[$i]}"
  echo "Starting run $((i+1))/${total}: ${cmd}"
  bash -lc "$cmd"
  code=$?

  if [ ${code} -ne 0 ]; then
    echo "Run $((i+1)) FAILED (exit ${code})."
    failed=$((failed+1))
  else
    echo "Run $((i+1)) succeeded."
    success=$((success+1))
  fi
done

echo "All runs finished. Success: ${success}, Failed: ${failed}, Total: ${total}"
