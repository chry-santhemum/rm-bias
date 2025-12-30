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
--student_model skywork-llama-8b \
--teacher_model claude-sonnet-4.5 \
--dataset chatgpt \
--topic_ids 0 1 3 4 6 7 8 9 10 \
--planner_type list_reverse \
--direction plus \
--n_new 8 \
--n_pop_initial 64 \
--n_pop_targets 16 8 8 \
--train_batch_sizes 8 8 16 \
--m_var 3 \
--n_planner_requests 32 \
--run_name "20251229-155635-list_reverse-chatgpt-plus"
