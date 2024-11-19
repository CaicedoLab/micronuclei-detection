#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import sys
import time
import torch
import skimage
import sklearn.metrics
import wandb
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import mnds
import mnmodel
import evaluation

CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/20X_c0_HeLa'
OUTPUT_DIR = "/model_output/output/"

# set CHTC writeable cahce directory for pytorch and matplotlib
os.environ['TORCH_HOME'] = CURRENT_PATH + '/.cache/torch'
os.environ['MPLCONFIGDIR'] = CURRENT_PATH + '/.cache/matplotlib/config'
torch.set_num_threads(8) # set only 8 cpus, the same number as requested

# Fixed Hyperparameters
PATCH_SIZE = 256
STRIDE = 8
FEATURE_SIZE = 384
TOKENS_PER_PATCH = PATCH_SIZE // STRIDE
STEP = 16
EPOCHS = 20
THRESHOLD = 0.5

LOSS_FN = 'combined'
LR = 1e-5
BATCH_SIZE = 32
FINETUNE = True
WEIGHT_DECAY = 1e-6

# Tunable Hyperparameters     
SCALE_FACTOR = 1.07
TEST_ANNOTATION_TYPE = 'edge' # for our data
# ANNOTATION_TYPE = 'filled' # for mnfinder data
ARCHITECTURE = "model version 2.1: evaluate on new HeLa images - 20241114 pilot validation"
DILATION = 0 # 2 might be the best, only used in prediction

if len(sys.argv) < 3:
    print("Use: prediction.py imidx gpu")
    sys.exit()


i = int(sys.argv[1])
gpu = sys.argv[2] # which gpu
device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'

# avoid files starting with . when untarring in CHTC
files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')]
# annot_files = [x for x in filelist if x.endswith('png')]
annot_files = [x for x in filelist if x.endswith('tif')] # no ground truth
annot_files.sort()

# Validate
predictions_dir = DIRECTORY + OUTPUT_DIR
models_dir = OUTPUT_DIR

# Load model and compute probabilities
model = mnmodel.MicronucleiModel(
    DIRECTORY, 
    device, 
    patch_size=PATCH_SIZE, 
    annotation_type=TEST_ANNOTATION_TYPE,
    edges=True, 
    gaussian=False # not needed in predict() function
)
# model.load(validation_file.replace('phenotype_outlines.png','pth'), model_dir=models_dir)
model.load('best_model_v2_1.pth', model_dir=models_dir)

# for i in tqdm(range(len(annot_files))):
if True:
    # Select image for analysis
    validation_file = annot_files[i]
    imid = validation_file.split('.')[0]
    
    
    key_file = open('./wandb_key.txt', 'r')
    key = key_file.readline()
    wandb.login(key=key)
    wandb.init(
        project='Best_Experiment',
        config={
            "architecture":ARCHITECTURE,
            "Loss": LOSS_FN,
            "Loss Weight": "all default, sam ratio (0.95focal+0.05dice) + gamma=2, etc",
            "fine_tuning":FINETUNE,
            "batch_size":BATCH_SIZE,
            "learning_rate":LR,
            "scale_factor":SCALE_FACTOR,
            # "test_annotation":TEST_ANNOTATION_TYPE,
            "epochs": EPOCHS,
            "feature_size":FEATURE_SIZE,
            "patch_size":PATCH_SIZE,
            "weight_decay":WEIGHT_DECAY,
            "probability_threshold":THRESHOLD,
            "dilation":DILATION,
            "gaussian":'gaussian not need for prediction'
        },
        name=f'{imid}'
    )
    
    # Load image and annotations
    filename_suffix = 'phenotype_HeLa.tif'
    im = mnds.read_image(DIRECTORY, imid, filename_suffix, scale=SCALE_FACTOR)
    
    if filename_suffix == 'phenotype_HeLa.tif': # add if the last channel == 3 !!!!!!!!!!!
        # 3 channel-iamge, concatenate
        im = np.mean(im, axis=2)
        
    im = np.array((im - np.min(im))/(np.max(im) - np.min(im)), dtype="float32")
    # mn_gt = mnds.read_micronuclei_masks(DIRECTORY, imid, SCALE_FACTOR, annotation_type=TEST_ANNOTATION_TYPE)
    

    probabilities = model.predict(im, stride=1, step=STEP, batch_size=BATCH_SIZE, dilation=DILATION)
    # filename = predictions_dir + validation_file.replace('phenotype_outlines.png','_probabilities')
    filename = predictions_dir + validation_file.replace(filename_suffix,'_probabilities') # no ground truth
    
    mn_pred = probabilities[0,:,:] > THRESHOLD
    # evaluation.segmentation_report(imid=imid, predictions=mn_pred, gt=mn_gt, intersection_ratio=0.1, report_obj='Micronuclei')
    np.save(filename, mn_pred)
    
    # release the resources
    torch.cuda.empty_cache()