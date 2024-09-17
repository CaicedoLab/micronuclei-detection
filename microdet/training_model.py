#!/usr/bin/env python
# coding: utf-8

'''
Train Best Model with all 18 images at once
'''

import os
import sys
import torch
import numpy as np
import mnds
import mnmodel
import evaluation
import wandb


CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/dataset_v2'
OUTPUT_DIR = "/model_output/output/"

# set CHTC writeable cahce directory for pytorch and matplotlib
os.environ['TORCH_HOME'] = CURRENT_PATH + '/.cache/torch'
os.environ['MPLCONFIGDIR'] = CURRENT_PATH + '/.cache/matplotlib/config'
torch.set_num_threads(8) # set only 8 cpus, the same number as requested

if len(sys.argv) < 2:
    print("Use: training_model.py gpu")
    sys.exit()

# gpu
gpu = sys.argv[1]
device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'

SCALE_FACTOR = 1.0
PATCH_SIZE = 256
STRIDE = 8
FEATURE_SIZE = 384
TOKENS_PER_PATCH = PATCH_SIZE // STRIDE
STEP = 16
EPOCHS = 20
THRESHOLD = 0.5
ANNOTATION_TYPE = 'edge' # train on our own data

LOSS_FN = 'combined'
LR = 1e-5
BATCH_SIZE = 32
FINETUNE = True
WEIGHT_DECAY = 1e-6

# Train
files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
annot_files = [x for x in filelist if x.endswith('png')]
annot_files.sort()
# annot_files = annot_files[0:10] # using all 18 images

training_files = annot_files.copy()

# read the txt files to pass the key, just do not pass private information into github
key_file = open('./wandb_key.txt', 'r')
key = key_file.readline()
wandb.login(key=key)
wandb.init(
    project='Best_Experiment',
    config={
        "architecture":"best model: train all 18 images",
        "Loss": LOSS_FN,
        "Loss Weight": "all default, sam ratio (0.95focal+0.05dice) + gamma=2, etc",
        "fine_tuning":FINETUNE,
        "batch_size":BATCH_SIZE,
        "learning_rate":LR,
        "epochs": EPOCHS,
        "feature_size":FEATURE_SIZE,
        "patch_size":PATCH_SIZE,
        "weight_decay":WEIGHT_DECAY,
        "probability_threshold":THRESHOLD
    },
    name=f'train_18images'
)

# Create model
model = mnmodel.MicronucleiModel(
    DIRECTORY, 
    device, 
    training_files=training_files, 
    validation_files=[], # input empty list when train with all 18 images of the best model
    patch_size=PATCH_SIZE,
    scale_factor=SCALE_FACTOR,
    edges=True
)

# Train
model.train(epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            learning_rate=LR, 
            loss_fn=LOSS_FN, 
            output_dir=OUTPUT_DIR, 
            finetune=FINETUNE,
            weight_decay=WEIGHT_DECAY
)

# Save
model.save(outdir=OUTPUT_DIR)

# release the resources
torch.cuda.empty_cache()
wandb.finish()
