#!/bin/bash

python train.py \
--dataset synthetic \
--runner_type evo \
--planner_type list_reverse \
--direction plus \
--n_new 8 \
--n_pop_initial 64


python train.py \
--dataset synthetic \
--runner_type evo \
--planner_type pair \
--direction plus \
--n_new 8 \
--n_pop_initial 64


python train.py \
--dataset synthetic \
--runner_type one_turn \
--planner_type list_reverse \
--direction plus \
--n_new 8 \
--n_pop_initial 128


python train.py \
--dataset synthetic \
--runner_type one_turn \
--planner_type pair \
--direction plus \
--n_new 8 \
--n_pop_initial 128