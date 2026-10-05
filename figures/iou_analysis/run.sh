#!/bin/bash

export HF_HUB_DISABLE_XET=1
python3 save_IoUs.py \
    --path /scr/data/annotated_mn_datasets/ \
    --save_path /scr/yren/IoUs/microSAM/ \
    --step 32 \
    --batch_size 4 \
    --prob_threshold 0.5 \
    --scale 1.0

