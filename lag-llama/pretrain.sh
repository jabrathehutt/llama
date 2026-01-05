#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run.py \
    -e "network_specialization_v1" \
    -r "experiments/results" \
    --single_dataset "network_telemetry" \
    --dataset_path "datasets" \
    --ckpt_path "lag-llama.ckpt" \
    --batch_size 64 \
    --max_epochs 50 \
    --num_batches_per_epoch 100 \
    --context_length 32 \
    --n_layer 8 \
    --n_head 9 \
    --n_embd_per_head 16 \
    --lr 0.0001 \
    --data_normalization "mean" \
    --gpu 0 \
    --wandb_mode "offline" \
    --lags_seq 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 60 72 84 96 108 120 132 144 156 168 180 192 204 216 228 240 252 264 276 288 300 312 324 336 348 360 372 384 396 408 420 432 444 456 468 480 492 504 516 528 540 552 564 576
