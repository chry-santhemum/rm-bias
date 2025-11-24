#!/bin/bash

python train.py \
--dataset synthetic_0 \
--runner_type evo \
--planner_type list \
--n_new 8 \
--n_pop_initial 128

python train.py \
--dataset synthetic_0 \
--runner_type evo \
--planner_type pair \
--n_new 8 \
--n_pop_initial 128

python train.py \
--dataset synthetic_0 \
--runner_type one_turn \
--planner_type list_reverse \
--n_new 16 \
--n_pop_initial 256
