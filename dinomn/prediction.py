#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import sys
import time
import skimage.morphology
import torch
import skimage
import sklearn.metrics
import wandb
from tqdm import tqdm

import numpy as np
# import matplotlib.pyplot as plt

from dinomn import mnds
from dinomn import mnmodel
from dinomn import evaluation

CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/all_data_micronuclei_no_rescale/validation'
OUTPUT_DIR = "/model_output/output/"

# set CHTC writeable cahce directory for pytorch and matplotlib
os.environ['TORCH_HOME'] = CURRENT_PATH + '/.cache/torch'
# os.environ['MPLCONFIGDIR'] = CURRENT_PATH + '/.cache/matplotlib/config'
torch.set_num_threads(8)

# Fixed Hyperparameters
PATCH_SIZE = 256
FEATURE_SIZE = 384
STEP = 64 # 64 is the best
EPOCHS = 20
THRESHOLD = 0.5
IoU_THRESHOLD = 0.1 # for micronuclei

LOSS_FN = 'combined'
LR = 1e-5
PREDICTION_BATCH = 4
FINETUNE = True
WEIGHT_DECAY = 1e-6

# Tunable Hyperparameters     
SCALE_FACTOR = 1 # Trained on non-scaled images
DILATION = 2 # the best, only used in prediction
ARCHITECTURE = f'Step{STEP} Experiment'

if len(sys.argv) < 2:
    print("Use: prediction.py gpu")
    sys.exit()

gpu = sys.argv[1] # which gpu
device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'

# avoid files starting with . when untarring in CHTC
files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')]
annot_files = [x for x in filelist if x.endswith('.phenotype_outlines.png')]
annot_files.sort()

# Validate
predictions_dir = DIRECTORY + OUTPUT_DIR
models_dir = OUTPUT_DIR

# Load model and compute probabilities
model = mnmodel.MicronucleiModel(
    data_dir=CURRENT_PATH + '/all_data_micronuclei_no_rescale/train',
    device=device
)
# model.load(validation_file.replace('phenotype_outlines.png','pth'), model_dir=models_dir)
model_name = 'DinoMN.pth'
model.load(model_path=f'/scr/yren/best_model/micronuclei_detection_project/{model_name}')


# Log in WanDB
key_file = open('./wandb_key.txt', 'r')
key = key_file.readline()
wandb.login(key=key)


for i in tqdm(range(len(annot_files))):
# if True:
    # Select image for analysis
    validation_file = annot_files[i]
    imid = validation_file.split('.')[0]
    
    wandb.init(
        project='Best_Experiment',
        config={
            "architecture":ARCHITECTURE,
            "Loss": LOSS_FN,
            "Loss Weight": "all default, sam ratio (0.95focal+0.05dice) + gamma=2, etc",
            "fine_tuning":FINETUNE,
            "prediction_batch_size":PREDICTION_BATCH,
            "learning_rate":LR,
            "scale_factor":'Predict on images that are not scaled',
            "epochs": EPOCHS,
            'step':STEP,
            "feature_size":FEATURE_SIZE,
            "patch_size":PATCH_SIZE,
            "weight_decay":WEIGHT_DECAY,
            "probability_threshold":THRESHOLD,
            "IoU_threshold":IoU_THRESHOLD,
            "dilation":DILATION,
            "gaussian":'gaussian not need for prediction',
            'Number of validation images':len(annot_files)
        },
        name=f'{imid}',
        reinit=True
    )
    
    # Load image and annotations
    im = mnds.read_image(DIRECTORY, imid, 'phenotype.tif', scale=SCALE_FACTOR)
        
    im = np.array((im - np.min(im))/(np.max(im) - np.min(im)), dtype="float32")
    mn_gt = mnds.read_image(DIRECTORY, imid, 'phenotype_outlines.png', scale=SCALE_FACTOR)
    mn_gt = mn_gt > 0 # convert to boolean (binary mask)
    
    # Document inference time
    s = time.time()
    probabilities = model.predict(im, stride=1, step=STEP, batch_size=PREDICTION_BATCH) # has model.eval() & with torch.no_grad()
    e = time.time()
    wandb.log({'Inference Time': e-s})
    filename = predictions_dir + validation_file.replace('phenotype_outlines.png','_probabilities')
    # filename = predictions_dir + validation_file.replace('phenotype_outlines.tif','_probabilities') # no ground truth case
    
    mn_pred = probabilities[0,:,:] > THRESHOLD
    labeled_mn = skimage.morphology.label(mn_pred)
    labeled_mn = np.asarray(labeled_mn, dtype='uint16') # if saving as img
    
    # dilate the labeled mn
    if DILATION > 0:
        dilation = skimage.morphology.disk(DILATION)
        labeled_mn = skimage.morphology.dilation(labeled_mn, dilation)
    
    evaluation.segmentation_report(imid=imid, predictions=labeled_mn, gt=mn_gt, intersection_ratio=IoU_THRESHOLD, report_obj='Micronuclei')
    
    # save labeled matrices
    np.save(filename, labeled_mn)
    
# release the resources
torch.cuda.empty_cache()