#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append('../')
import torch
import numpy as np
import wandb
import dinomn.mnmodel as mnmodel


CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/annotated_mn_datasets/'
OUTPUT_DIR = "model_output/"

if not os.path.exists(DIRECTORY + OUTPUT_DIR):
    os.makedirs(DIRECTORY + OUTPUT_DIR)

# set CHTC writeable cahce directory for pytorch and matplotlib
os.environ['TORCH_HOME'] = CURRENT_PATH + '/.cache/torch'
# os.environ['MPLCONFIGDIR'] = CURRENT_PATH + '/.cache/matplotlib/config'
torch.set_num_threads(8)

if len(sys.argv) < 2:
    print("Use: training_model.py gpu")
    sys.exit()

# gpu
gpu = sys.argv[1]
device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'

# Fixed Hyperparameters
PATCH_SIZE = 256
FEATURE_SIZE = 384
EPOCHS = 20
THRESHOLD = 0.5

LOSS_FN = 'combined'
LR = 1e-5
BATCH_SIZE = 4 # best training batch size
FINETUNE = True
WEIGHT_DECAY = 1e-6

# Tunable Hyperparameters
SCALE_FACTOR = 1.0
GAUSSIAN = True
EDGES = False
ARCHITECTURE = "DinoMN: test new retraining code"

OVERSAMPLE = 'Yes' # if Yes, save model name as DinoMN_oversampled

num_training_files = len(os.listdir(DIRECTORY + 'train/images'))
num_validation_files = len(os.listdir(DIRECTORY + 'validation/images'))
wandb.init(
    project='Best_Experiment',
    config={
        "architecture":ARCHITECTURE,
        "Loss": LOSS_FN,
        "Loss Weight": "all default, sam ratio (0.95focal+0.05dice) + gamma=2, etc",
        "fine_tuning":FINETUNE,
        "training_batch_size":BATCH_SIZE,
        "start_learning_rate":LR,
        "lr_scheduler":"Cosine",
        "scale_factor":'Trained on non-scaled images',
        "epochs": EPOCHS,
        "feature_size":FEATURE_SIZE,
        "patch_size":PATCH_SIZE,
        "weight_decay":WEIGHT_DECAY,
        "probability_threshold":THRESHOLD,
        "gaussian":GAUSSIAN,
        'edges':EDGES,
        'Number of training images':num_training_files,
        'Number of validation images':num_validation_files,
        'Oversample':OVERSAMPLE
    },
    name=f'model version 3 (test new predict())'
)

# Create model
model = mnmodel.MicronucleiModel(
    device=device,
    data_dir=DIRECTORY,
    patch_size=PATCH_SIZE,
    scale_factor=SCALE_FACTOR,
    edges=EDGES, # False, this will recover the input edges, reducing performance
    gaussian=GAUSSIAN
)

# Train
model.train(epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            learning_rate=LR, 
            loss_fn=LOSS_FN, 
            finetune=FINETUNE,
            weight_decay=WEIGHT_DECAY,
            wandb_mode=True
)


# Save
model.save(outdir=OUTPUT_DIR, model_name='DinoMN')

# release the resources
torch.cuda.empty_cache()
wandb.finish()
