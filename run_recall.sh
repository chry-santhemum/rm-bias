#!/bin/bash
# Run recall experiments with synthetic biases

# # remaining ones
# python recall_exp.py \
#     --bias_type affirm \
#     --topic_id 0 \
#     --num_seeds 10 \
#     --bias_strength 3.0 \
#     --noise_strength 3.0 \
#     --random_seed 42

# python recall_exp.py \
#     --bias_type affirm \
#     --topic_id 0 \
#     --num_seeds 5 \
#     --bias_strength 3.0 \
#     --noise_strength 5.0 \
#     --random_seed 47

# python recall_exp.py \
#     --bias_type affirm \
#     --topic_id 0 \
#     --num_seeds 5 \
#     --bias_strength 3.0 \
#     --noise_strength 7.0 \
#     --random_seed 47

# python recall_exp.py \
#     --bias_type affirm \
#     --topic_id 0 \
#     --num_seeds 5 \
#     --bias_strength 3.0 \
#     --noise_strength 10.0 \
#     --random_seed 47

# python recall_exp.py \
#     --bias_type affirm \
#     --topic_id 0 \
#     --num_seeds 10 \
#     --bias_strength 3.0 \
#     --noise_strength 17.0 \
#     --random_seed 47

python recall_exp.py \
    --bias_type affirm \
    --topic_id 0 \
    --num_seeds 10 \
    --bias_strength 3.0 \
    --noise_strength 27.0 \
    --random_seed 47

# various seeds

# python recall_exp.py \
#     --bias_type affirm \
#     --topic_id 5 \
#     --num_seeds 10 \
#     --bias_strength 3.0 \
#     --noise_strength 3.0

# python recall_exp.py \
#     --bias_type affirm \
#     --topic_id 9 \
#     --num_seeds 10 \
#     --bias_strength 3.0 \
#     --noise_strength 3.0

# python recall_exp.py \
#     --bias_type headers \
#     --topic_id 0 \
#     --num_seeds 10 \
#     --bias_strength 3.0 \
#     --noise_strength 3.0

# python recall_exp.py \
#     --bias_type headers \
#     --topic_id 3 \
#     --num_seeds 10 \
#     --bias_strength 3.0 \
#     --noise_strength 3.0

# python recall_exp.py \
#     --bias_type headers \
#     --topic_id 5 \
#     --num_seeds 10 \
#     --bias_strength 3.0 \
#     --noise_strength 3.0

# python recall_exp.py \
#     --bias_type list \
#     --topic_id 2 \
#     --num_seeds 10 \
#     --bias_strength 3.0 \
#     --noise_strength 3.0

# python recall_exp.py \
#     --bias_type list \
#     --topic_id 3 \
#     --num_seeds 10 \
#     --bias_strength 3.0 \
#     --noise_strength 3.0

# python recall_exp.py \
#     --bias_type list \
#     --topic_id 6 \
#     --num_seeds 10 \
#     --bias_strength 3.0 \
#     --noise_strength 3.0

# python recall_exp.py \
#     --bias_type list \
#     --topic_id 8 \
#     --num_seeds 10 \
#     --bias_strength 3.0 \
#     --noise_strength 3.0

python recall_exp.py \
    --bias_type list \
    --topic_id 0 \
    --num_seeds 10 \
    --bias_strength 3.0 \
    --noise_strength 3.0

python recall_exp.py \
    --bias_type list \
    --topic_id 1 \
    --num_seeds 10 \
    --bias_strength 3.0 \
    --noise_strength 3.0
