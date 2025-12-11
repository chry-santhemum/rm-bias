#!/bin/bash

python train.py \
--dataset synthetic \
--student_model skywork-llama-8b-exp \
--teacher_model skywork-llama-8b \
--planner_type pair \
--direction plus \
--n_new 8 \
--n_pop_initial 64 \
--m_var 3 \
--n_planner_requests 64 \
--n_baseline_rollouts 16 \
--n_rewrite_rollouts 8

python train.py \
--dataset synthetic \
--student_model skywork-llama-8b-exp \
--teacher_model skywork-llama-8b \
--planner_type list_reverse \
--direction plus \
--n_new 8 \
--n_pop_initial 64 \
--m_var 3 \
--n_planner_requests 64 \
--n_baseline_rollouts 16 \
--n_rewrite_rollouts 8

python train.py \
--dataset synthetic \
--student_model skywork-qwen-0.6b \
--teacher_model skywork-llama-8b \
--planner_type pair \
--direction plus \
--n_new 8 \
--n_pop_initial 64 \
--m_var 3 \
--n_planner_requests 64 \
--n_baseline_rollouts 16 \
--n_rewrite_rollouts 8

python train.py \
--dataset synthetic \
--student_model skywork-llama-8b \
--teacher_model skywork-llama-8b-exp \
--planner_type pair \
--direction plus \
--n_new 8 \
--n_pop_initial 64 \
--m_var 3 \
--n_planner_requests 64 \
--n_baseline_rollouts 16 \
--n_rewrite_rollouts 8