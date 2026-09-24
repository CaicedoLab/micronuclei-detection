#!/bin/bash

# python3 cpsam_prediction.py \
#     --gpu 0 \
#     --train_path /scr/data/annotated_mn_datasets/train/images \
#     --save_path /hdd/jcaicedo/projects/micronuclei_detection/Train_and_Eval/mndino_data/baselines/cpsam_fintuned_predictions \
#     --finetune \
#     --iou_threshold 0.1 \
#     --wandb_mode

python3 microsam_prediction.py \
    --gpu 1 \
    --train_path /scr/yren/microsam_data/train/ \
    --pred_path /scr/data/annotated_mn_datasets/test/images/ \
    --save_path /scr/yren/microsam_predictions/ \
    --finetune \
    --wandb_mode