#!/bin/bash

python train.py \
--dataset synthetic \
--runner_type evo \
--planner_type list_reverse \
--direction plus \
--n_new 8 \
--n_pop_initial 64