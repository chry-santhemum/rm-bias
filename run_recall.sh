#!/bin/bash
# Run recall experiments with synthetic biases

python recall_exp.py \
    --bias_type affirm \
    --topic_id 0 \
    --num_seeds 3 \
    --bias_strength 3.0 \
    --noise_strength 10.0

# python recall_exp.py \
#     --bias_type affirm \
#     --topic_id 0 \
#     --num_seeds 10 \
#     --bias_strength 3.0 \
#     --noise_strength 7.0


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
