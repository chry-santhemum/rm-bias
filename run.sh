#!/bin/bash

python train.py --dataset synthetic_0 --runner_type evo --planner_type list
python train.py --dataset synthetic_0 --runner_type evo --planner_type pair
python train.py --dataset synthetic_0 --runner_type one_turn --planner_type list --n_pop_initial 256
# python train.py --dataset synthetic_0 --runner_type one_turn --planner_type pair --n_pop_initial 256
