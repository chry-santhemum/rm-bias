#!/bin/bash

python train.py \
--dataset synthetic_0 \
--runner_type evo \
--planner_type pair \
--direction plus \
--n_new 8 \
--n_pop_initial 64 \
--run_name 20251128-145500-pair-synthetic_0-plus