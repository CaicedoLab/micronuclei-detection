#!/bin/bash

python3 training_model.py --path '/scr/yren/annotated_mn_datasets/' --gpu 3 --epochs 20 --loss_fn 'combined' --lr 1e-6 --scale 1.0 --gaussian --wandb_mode

python3 prediction.py --path '/scr/yren/annotated_mn_datasets/' --gpu 3 --step 64 --batch_size 4 --prob_threshold 0.5 --iou_threshold 0.1 --scale 1 --wandb_mode