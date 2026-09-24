#!/bin/bash

python3 training_model.py \
    --path /scr/data/annotated_mn_datasets/ \
    --gpu 0 \
    --model_type dino \
    --epochs 20 \
    --loss_fn combined \
    --lr 1e-6 \
    --scale 1.0 \
    --gaussian \
    --wandb_mode

python3 prediction.py \
    --path '/scr/data/annotated_mn_datasets/'\
    --test_set \
    --gpu 0 \
    --model_type dino \
    --step 32 \
    --batch_size 4 \
    --prob_threshold 0.5 \
    --iou_threshold 0.1 \
    --scale 1 \
    --wandb_mode