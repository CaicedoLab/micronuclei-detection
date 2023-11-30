#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import mnmodel

# In[3]:

SCALE_FACTOR = 1.0

PATCH_SIZE = 256
STRIDE = 8
FEATURE_SIZE = 384
TOKENS_PER_PATCH = PATCH_SIZE // STRIDE
DIRECTORY = "/home/caicedo/scr/jcaicedo/Micronuclei-data/dataset_v2/"
OUTPUT_DIR = "experiments/2023-11-12C/models/"

EPOCHS = 20
BATCH_SIZE = 48
LR = 0.01

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# In[4]:


filelist = os.listdir(DIRECTORY)
annot_files = [x for x in filelist if x.endswith('png')]


# In[5]:


for i in range(len(annot_files)):

    # Leave-one-out split
    training_files = annot_files.copy()
    validation_files = [annot_files[i]]
    del training_files[i]
    
    print(" *** ", validation_files, " *** ")

    # Create model
    model = mnmodel.MicronucleiModel(DIRECTORY, device, training_files=training_files, validation_files=validation_files, scale_factor=SCALE_FACTOR)
    
    # Train
    model.train(EPOCHS, BATCH_SIZE, LR)
    
    # Validate
    model.validate()
    
    # Save
    model.save(outdir=OUTPUT_DIR)





