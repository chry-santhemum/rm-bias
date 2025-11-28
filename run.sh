#!/bin/bash

python train.py \
--dataset synthetic_0 \
--runner_type one_turn \
--planner_type pair \
--direction plus \
--n_new 8 \
--n_pop_initial 128 \
--run_name 20251128-091719-pair-synthetic_0-plus


python train.py \
--dataset synthetic_0 \
--runner_type evo \
--planner_type pair \
--direction plus \
--n_new 8 \
--n_pop_initial 64