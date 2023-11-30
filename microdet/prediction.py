#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
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
FEATURE_SIZE = 384
TOKENS_PER_PATCH = PATCH_SIZE // STRIDE
STEP = 16

DIRECTORY = "/home/caicedo/scr/jcaicedo/Micronuclei-data/dataset_v2/"

BATCH_SIZE = 512
EPOCHS = 20
LR = 0.01

THRESHOLD = 0.25

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'


# In[3]:


filelist = os.listdir(DIRECTORY)
annot_files = [x for x in filelist if x.endswith('png')]

predictions_dir = DIRECTORY + "experiments/2023-11-08/predictions/"
models_dir = "experiments/2023-11-08/models/"


# In[4]:


for i in range(len(annot_files)):
    # Select image for analysis
    validation_file = annot_files[i]
    imid = validation_file.split('.')[0]
    print(imid)
    
    # Load image and annotations
    im = mnds.read_image(DIRECTORY, imid, 'phenotype.tif', scale=SCALE_FACTOR)
    im = skimage.exposure.rescale_intensity(im, out_range=np.float32)
    gt = mnds.read_micronuclei_annotations(DIRECTORY, imid)
    
    # Load model and compute probabilities
    model = mnmodel.MicronucleiModel(DIRECTORY, device)
    model.load(validation_file.replace('phenotype_outlines.png','pth'), model_dir=models_dir)
    probabilities = model.predict(im, stride=STRIDE, patch_size=PATCH_SIZE, step=STEP, batch_size=BATCH_SIZE)
    
    # Run evaluations
    results = evaluation.prediction_report(imid, probabilities, gt, THRESHOLD, predictions_dir)
    evaluation.display_detections(im, imid, results, predictions_dir)


# In[ ]:




