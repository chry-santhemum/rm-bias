#!/bin/bash

# MODEL_PAIRS=(
#     "skywork-llama-8b-exp,skywork-llama-8b"
#     "skywork-qwen-0.6b,skywork-llama-8b"
#     "skywork-llama-8b,claude-sonnet-4.5"
# )

# PLANNER_TYPES=("pair" "list_reverse")

# for pair in "${MODEL_PAIRS[@]}"; do
#     IFS=',' read -r student teacher <<< "$pair"
#     for planner in "${PLANNER_TYPES[@]}"; do
#         echo "=== Running: $planner with $student / $teacher ==="
#         python train.py \
#             --dataset synthetic \
#             --student_model "$student" \
#             --teacher_model "$teacher" \
#             --planner_type "$planner" \
#             --direction plus \
#             --n_new 16 \
#             --n_pop_initial 128 \
#             --m_var 3 \
#             --n_planner_requests 64 \
#             --n_baseline_rollouts 16 \
#             --n_rewrite_rollouts 4
#     done
# done



python train.py \
--dataset clio \
--student_model skywork-llama-8b \
--teacher_model claude-sonnet-4.5 \
--planner_type pair \
--direction plus \
--n_new 8 \
--n_pop_initial 64 \
--n_pop_targets 16 8 \
--train_batch_sizes 8 8 \
--m_var 2 \
--n_planner_requests 32



python train.py \
--dataset clio \
--student_model skywork-llama-8b \
--teacher_model claude-sonnet-4.5 \
--planner_type list_reverse \
--direction plus \
--n_new 8 \
--n_pop_initial 64 \
--n_pop_targets 16 8 \
--train_batch_sizes 8 8 \
--m_var 2 \
--n_planner_requests 32


python train.py \
--dataset clio \
--student_model skywork-llama-8b \
--teacher_model claude-sonnet-4.5 \
--planner_type list \
--direction plus \
--n_new 8 \
--n_pop_initial 64 \
--n_pop_targets 16 8 \
--train_batch_sizes 8 8 \
--m_var 2 \
--n_planner_requests 32

