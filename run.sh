#!/bin/bash

python train.py \
--dataset synthetic \
--planner_type pair \
--direction plus \
--n_new 8 \
--n_pop_initial 64 \
--m_var 3 \
--n_planner_requests 64 \
--n_baseline_rollouts 16 \
--n_rewrite_rollouts 8
