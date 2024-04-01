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


# In[2]:


SCALE_FACTOR = 1.0

PATCH_SIZE = 256
STRIDE = 8
FEATURE_SIZE = 384 # dinov2 vit small model
# FEATURE_SIZE = 768 # dinov2 vit base model
TOKENS_PER_PATCH = PATCH_SIZE // STRIDE
STEP = 16

# DIRECTORY = "/dgx1nas1/storage/data/jcaicedo/micronuclei/data/dataset_v2/"

CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/dataset_v2'

BATCH_SIZE = 32
EPOCHS = 20
LR = 0.0001

THRESHOLD = 0.25
# THRESHOLD = 0.5

# set CHTC writeable cahce directory for pytorch and matplotlib
os.environ['TORCH_HOME'] = CURRENT_PATH + '/.cahce/torch'
os.environ['MPLCONFIGDIR'] = CURRENT_PATH + '/.cache/matplotlib/config'

if len(sys.argv) < 3:
    print("Use: prediction.py imidx gpu")
    sys.exit()

i = int(sys.argv[1])
gpu = sys.argv[2]
device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'


# In[3]:


# avoid files starting with . when untarring in CHTC
files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')]
annot_files = [x for x in filelist if x.endswith('png')]
annot_files.sort()


# expriment directory with dinov2 small
predictions_dir = DIRECTORY + "/model_output/nuclei_experiments/"
models_dir = "/model_output/nuclei_experiments/" 

# In[4]:

# Initiate Weights and Biases
# wandb.login(key='b3f4f9254c123781af918799b27affa92d8f4eeb')
# wandb.init(
#     project='micronuclei-segmentation-prediction',
    
#     # hyperparameters
#     config={
#         "architecture":"base_model: 3 blocks of 2x2 transposed conv and 3x3 conv layers",
#         "learning_rate":LR,
#         "threshold":THRESHOLD,
#         "epochs": EPOCHS,
#         "feature_size":FEATURE_SIZE,
#         "batch_size":BATCH_SIZE,
#         "patch_size":PATCH_SIZE,
#         "fine_tuning":False,
#         "Scheduler":"Cos scheduler"
#     }
# )

#for i in range(len(annot_files)):
if True:
    # Select image for analysis
    validation_file = annot_files[i]
    imid = validation_file.split('.')[0]
    
    # Load image and annotations
    im = mnds.read_image(DIRECTORY, imid, 'phenotype.tif', scale=SCALE_FACTOR)
    im = np.array((im - np.min(im))/(np.max(im) - np.min(im)), dtype="float32")
    #im = skimage.exposure.rescale_intensity(im, out_range=np.float32)
    mn_gt = mnds.read_micronuclei_masks(DIRECTORY, imid, SCALE_FACTOR)
    n_gt = mnds.read_nuclei_masks(DIRECTORY, imid, SCALE_FACTOR)
    
    # Load model and compute probabilities
    model = mnmodel.MicronucleiModel(DIRECTORY, device, patch_size=PATCH_SIZE, edges=True)
    model.load(validation_file.replace('phenotype_outlines.png','pth'), model_dir=models_dir)
    probabilities = model.predict(im, stride=1, step=STEP, batch_size=BATCH_SIZE)
    filename = predictions_dir + validation_file.replace('phenotype_outlines.png','_probabilities')
    np.save(filename, probabilities)
    
    # Calculate the jaccard score between probabilities and gt
    # print(f'probabilities shape(expected (2,2960,2960)): {probabilities.shape} \n')

    # Run evaluations
    mn_pred = probabilities[0,:,:] > THRESHOLD
    n_pred = probabilities[1,:,:] > THRESHOLD
    
    # computationally demanding, only report for micronuclei for efficiency purpose
    evaluation.segmentation_report(imid=imid, predictions=mn_pred, gt=mn_gt, intersection_ratio=0.1, report_obj='Micronuclei')
    # evaluation.segmentation_report(predictions=n_pred, gt=n_gt, intersection_ratio=0.5, report_obj='Nuclei')
    
    #results = evaluation.prediction_report(imid, probabilities, gt, THRESHOLD, predictions_dir)
    #evaluation.display_detections(im, imid, results, predictions_dir)

# In[ ]:

# wandb.finish()


