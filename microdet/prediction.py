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

import numpy as np
import matplotlib.pyplot as plt

import mnds
import mnmodel
import evaluation

CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/dataset_v2'
OUTPUT_DIR = "/model_output/output/"

# set CHTC writeable cahce directory for pytorch and matplotlib
os.environ['TORCH_HOME'] = CURRENT_PATH + '/.cache/torch'
os.environ['MPLCONFIGDIR'] = CURRENT_PATH + '/.cache/matplotlib/config'
torch.set_num_threads(8) # set only 8 cpus, the same number as requested

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

if len(sys.argv) < 2:
    print("Use: prediction.py gpu")
    sys.exit()


# i = int(sys.argv[1])
gpu = sys.argv[1] # which gpu
device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'

# avoid files starting with . when untarring in CHTC
files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')]
annot_files = [x for x in filelist if x.endswith('png')]
annot_files.sort()

# Validate
predictions_dir = DIRECTORY + OUTPUT_DIR
models_dir = OUTPUT_DIR

# Predict images without ground truth, comment out code of prediciting with ground truth
# Load model and compute probabilities
model = mnmodel.MicronucleiModel(DIRECTORY, device, patch_size=PATCH_SIZE, edges=True)
# model.load(validation_file.replace('phenotype_outlines.png','pth'), model_dir=models_dir)
model.load('best_model.pth', model_dir=models_dir)

for i in range(0, len(annot_files)):
    # Select image for analysis
    validation_file = annot_files[i]
    imid = validation_file.split('.')[0]
    
    # Load image and annotations
    im = mnds.read_image(DIRECTORY, imid, 'phenotype.tif', scale=SCALE_FACTOR)
    im = np.array((im - np.min(im))/(np.max(im) - np.min(im)), dtype="float32")
    mn_gt = mnds.read_micronuclei_masks(DIRECTORY, imid, SCALE_FACTOR) # no ground truth in plates
    

    probabilities = model.predict(im, stride=1, step=STEP, batch_size=BATCH_SIZE)
    filename = predictions_dir + validation_file.replace('phenotype_outlines.png','_probabilities')
    np.save(filename, probabilities)
    
    mn_pred = probabilities[0,:,:] > THRESHOLD
    evaluation.segmentation_report(imid=imid, predictions=mn_pred, gt=mn_gt, intersection_ratio=0.1, report_obj='Micronuclei')
    
    # release the resources
    torch.cuda.empty_cache()