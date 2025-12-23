#!/bin/bash

for j in {1..10}; do
    echo "Running iteration $j"

    python3 training_model.py \
        --path '/scr/data/annotated_mn_datasets/' \
        --gpu 0 \
        --epochs 20 \
        --batch_size 4 \
        --loss_fn 'combined' \
        --lr 1e-5 \
        --scale 1.0 \
        --gaussian \
        --wandb_mode \
        --tags "mndino exp on cleaned data - iter $j" \
        --iteration $j

    python3 prediction.py \
        --path '/scr/data/annotated_mn_datasets/' \
        # --test_set \ run on validation set for model selection
        --gpu 0 \
        # --step 32 \ use 64 step for 10 iteration validation set
        --step 64
        --batch_size 4 \
        --prob_threshold 0.5 \
        --iou_threshold 0.1 \
        --scale 1 \
        --wandb_mode \
        --tags "mndino exp on cleaned data - iter $j" \
        --iteration $j
done