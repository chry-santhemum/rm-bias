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
--student_model skywork-qwen-8b \
--teacher_model claude-sonnet-4.5 \
--dataset handpick \
--topic_ids 0 1 \
--planner_type list_reverse \
--direction plus \
--n_new 8 \
--n_pop_initial 64 \
--n_pop_targets 16 10 8  \
--train_batch_sizes 16 32 32 \
--m_var 3 \
--n_planner_requests 32 \
--val_split_size 64


# python train.py \
# --student_model skywork-llama-8b \
# --teacher_model claude-sonnet-4.5 \
# --dataset handpick \
# --topic_ids 12 13 14 17 18 19 \
# --planner_type list_reverse \
# --direction plus \
# --n_new 8 \
# --n_pop_initial 64 \
# --n_pop_targets 16 12 12 8 \
# --train_batch_sizes 16 32 32 32 \
# --m_var 2 \
# --n_planner_requests 32 \
# --n_rewrite_rollouts 2 \
# --n_validate_rollouts 1 \
# --judge_train_first_n_rollouts 1 \
# --judge_train_first_n_user_prompts 32 \
# --judge_val_first_n_rollouts 2 \
# --judge_val_first_n_user_prompts 32 \
# --val_split_size 32 \
# --run_name "20251231-034737-list_reverse-handpick-plus" \
# --baseline_path "20251231-014058-list_reverse-handpick-plus" \
# --start_from 2



# original: (64 + 32 + 32)*16
# new: (48+48+36)
