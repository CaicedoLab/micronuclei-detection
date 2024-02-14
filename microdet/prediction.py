#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import sys
import time
import torch
import skimage

import numpy as np
import matplotlib.pyplot as plt

import mnds
import mnmodel
import evaluation


# In[2]:


SCALE_FACTOR = 1.0

PATCH_SIZE = 256
STRIDE = 8
# FEATURE_SIZE = 384 # dinov2 vit small model
FEATURE_SIZE = 768 # dinov2 vit base model
TOKENS_PER_PATCH = PATCH_SIZE // STRIDE
STEP = 16

# DIRECTORY = "/dgx1nas1/storage/data/jcaicedo/micronuclei/data/dataset_v2/"

CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/dataset_v2/'

# set CHTC writeable cahce directory for pytorch and matplotlib
os.environ['TORCH_HOME'] = CURRENT_PATH + '/.cahce/torch'
os.environ['MPLCONFIGDIR'] = CURRENT_PATH + '/.cache/matplotlib/config'

BATCH_SIZE = 480
EPOCHS = 20
LR = 0.01

THRESHOLD = 0.25

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

# predictions_dir = DIRECTORY + "experiments/2024-01-31A/predictions/"
# models_dir = "experiments/2024-01-31A/models/"


predictions_dir = DIRECTORY + "/model_output/reproduced_base_model/"
models_dir = "/model_output/reproduced_base_model/" # under microdet folder in CHTC, mnmodel will load the DIRECTORY which leads to dataset_v2 folder


# In[4]:


#for i in range(len(annot_files)):
if True:
    # Select image for analysis
    validation_file = annot_files[i]
    imid = validation_file.split('.')[0]
    print(imid)
    
    # Load image and annotations
    im = mnds.read_image(DIRECTORY, imid, 'phenotype.tif', scale=SCALE_FACTOR)
    im = np.array((im - np.min(im))/(np.max(im) - np.min(im)), dtype="float32")
    #im = skimage.exposure.rescale_intensity(im, out_range=np.float32)
    gt = mnds.read_micronuclei_annotations(DIRECTORY, imid)
    
    # Load model and compute probabilities
    model = mnmodel.MicronucleiModel(DIRECTORY, device)
    model.load(validation_file.replace('phenotype_outlines.png','pth'), model_dir=models_dir)
    probabilities = model.predict(im, stride=STRIDE, patch_size=PATCH_SIZE, step=STEP, batch_size=BATCH_SIZE)
    filename = predictions_dir + validation_file.replace('phenotype_outlines.png','_probabilities')
    np.save(filename, probabilities)
    
    # Run evaluations
    results = evaluation.prediction_report(imid, probabilities, gt, THRESHOLD, predictions_dir)
    evaluation.display_detections(im, imid, results, predictions_dir)


# In[ ]:




