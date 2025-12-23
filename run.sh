!/bin/bash

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
        ...
    done
done


python train.py \
--student_model recall-affirm \
--teacher_model skywork-llama-8b \
--dataset handpick \
--topic_ids 7 \
--planner_type list_reverse \
--direction plus \
--n_new 8 \
--n_pop_initial 64 \
--n_pop_targets 16 8 \
--train_batch_sizes 10 10 \
--m_var 2 \
--n_planner_requests 64
