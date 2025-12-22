#!/bin/bash

# for j in {1..10}; do
    # echo "Running iteration $j"

python3 training_model.py \
    --path '/scr/yren/annotated_mn_datasets/' \
    --gpu 0 \
    --epochs 20 \
    --batch_size 4 \
    --loss_fn 'combined' \
    --lr 1e-5 \
    --scale 1.0 \
    --gaussian \
    --wandb_mode \

python3 prediction.py \
    --path '/scr/yren/annotated_mn_datasets/' \
    --test_set \
    --gpu 0 \
    --step 32 \
    --batch_size 4 \
    --prob_threshold 0.5 \
    --iou_threshold 0.1 \
    --scale 1 \
    --wandb_mode \

# done