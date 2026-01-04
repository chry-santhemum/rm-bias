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
--dataset handpick \
--topic_ids 2 4 5 \
--planner_type list_reverse \
--direction plus \
--n_new 8 \
--n_pop_initial 64 \
--n_pop_targets 12 10 10 10 8  \
--train_batch_sizes 8 16 16 16 16 \
--m_var 6 \
--n_planner_requests 32 \
--val_split_size 32 \
--context all
