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
#         ...
#     done
# done

python train.py \
--student_model skywork-llama-8b \
--teacher_model claude-sonnet-4.5 \
--dataset chatgpt \
--topic_ids 0 \
--planner_type list_reverse \
--direction plus \
--n_new 8 \
--n_pop_initial 64 \
--n_pop_targets 16 16 8 8  \
--train_batch_sizes 8 16 16 32 \
--m_var 3 \
--n_planner_requests 32 \
--val_split_size 0 \
--context vanilla
