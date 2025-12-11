#!/bin/bash

# python train.py \
# --dataset synthetic \
# --planner_type pair \
# --direction plus \
# --n_new 8 \
# --n_pop_initial 64 \
# --m_var 3 \
# --n_planner_requests 64 \
# --n_baseline_rollouts 16 \
# --n_rewrite_rollouts 8

python train.py \
--dataset synthetic \
--student_model skywork-llama-8b \
--teacher_model skywork-llama-8b-exp \
--planner_type pair \
--direction plus \
--n_new 4 \
--n_pop_initial 8 \
--m_var 1 \
--n_planner_requests 8 \
--n_baseline_rollouts 8 \
--n_rewrite_rollouts 4