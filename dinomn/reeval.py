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

from dinomn import mnds
from dinomn import mnmodel
from dinomn import evaluation


# In[2]:


SCALE_FACTOR = 1.0

PATCH_SIZE = 256
STRIDE = 8
FEATURE_SIZE = 384
TOKENS_PER_PATCH = PATCH_SIZE // STRIDE
STEP = 16

DIRECTORY = "/dgx1nas1/storage/data/jcaicedo/micronuclei/data/dataset_v2/"

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


filelist = os.listdir(DIRECTORY)
annot_files = [x for x in filelist if x.endswith('png')]
annot_files.sort()

predictions_dir = DIRECTORY + "experiments/2024-01-31B/predictions/"


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
    gt = mnds.read_micronuclei_annotations(DIRECTORY, imid)
    
    # Load predictions
    filename = predictions_dir + validation_file.replace('phenotype_outlines.png','_probabilities.npy')
    probabilities = np.load(filename)
    print(probabilities.shape)
    print("Detections: ",np.sum(probabilities > THRESHOLD))
    loc = np.where(probabilities > THRESHOLD)
    masks = np.zeros((probabilities.shape[0]*8, probabilities.shape[1]*8), dtype="uint8")
    for i in range(len(loc[0])):
        r = loc[0][i]*8
        c = loc[1][i]*8
        masks[r-4:r+4,c-4:c+4] = i + 1 
    skimage.io.imsave(filename.replace('_probabilities.npy','_detections.png'), masks)
    
    # Run evaluations
    results = evaluation.prediction_report(imid, probabilities, gt, THRESHOLD, predictions_dir)
    evaluation.display_detections(im, imid, results, predictions_dir)


# In[ ]:




