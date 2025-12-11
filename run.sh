#!/bin/bash

# Model pairs: student_model,teacher_model
MODEL_PAIRS=(
    "skywork-llama-8b-exp,skywork-llama-8b"
    "skywork-qwen-0.6b,skywork-llama-8b"
    "skywork-llama-8b,claude-sonnet-4.5"
)

PLANNER_TYPES=("pair" "list_reverse")

for pair in "${MODEL_PAIRS[@]}"; do
    IFS=',' read -r student teacher <<< "$pair"
    for planner in "${PLANNER_TYPES[@]}"; do
        echo "=== Running: $planner with $student / $teacher ==="
        python train.py \
            --dataset synthetic \
            --student_model "$student" \
            --teacher_model "$teacher" \
            --planner_type "$planner" \
            --direction plus \
            --n_new 16 \
            --n_pop_initial 128 \
            --m_var 3 \
            --n_planner_requests 64 \
            --n_baseline_rollouts 16 \
            --n_rewrite_rollouts 8
    done
done
