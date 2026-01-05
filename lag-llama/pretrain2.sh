#!/bin/bash

mkdir -p experiments/results

EXP_NAME="network_pretraining_real_world"
SEED=42

# Ensure -d points to the folder containing 'clean_training_traffic_data.csv'
python run2.py \
    -e $EXP_NAME \
    -d "." \
    --seed $SEED \
    -r "experiments/results" \
    --batch_size 256 \
    --max_epochs 50 \
    --num_batches_per_epoch 100 \
    --lr 0.0001 \
    --context_length 32 \
    --n_layer 8 \
    --n_head 9 \
    --n_embd_per_head 16 \
    --data_normalization "mean" \
    --gpu 0
