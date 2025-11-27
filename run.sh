#!/bin/bash

python train.py \
--dataset synthetic_0 \
--runner_type evo \
--planner_type pair \
--n_new 8 \
--n_pop_initial 128
