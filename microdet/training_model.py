#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import torch
import mnmodel

# In[3]:

SCALE_FACTOR = 1.0

PATCH_SIZE = 256
STRIDE = 8
FEATURE_SIZE = 384 # dinov2 vit small model
# FEATURE_SIZE = 768 # dinov2 vit base model
TOKENS_PER_PATCH = PATCH_SIZE // STRIDE

# Reconstructing path in CHTC
CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/dataset_v2'
OUTPUT_DIR = "/model_output/pixel_decoder2/"

# set CHTC writeable cahce directory
os.environ['TORCH_HOME'] = CURRENT_PATH + '/.cahce/torch'

EPOCHS = 20
BATCH_SIZE = 32
LR = 0.0001

if len(sys.argv) < 3:
    print("Use: python training_model.py imidx gpu")
    sys.exit()

i = int(sys.argv[1])
gpu = sys.argv[2]
device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'


# In[4]:


# avoid files starting with . when untarring in CHTC
files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')]
annot_files = [x for x in filelist if x.endswith('png')]
annot_files.sort()
annot_files = annot_files[0:10]

# In[5]:


#for i in range(len(annot_files)):
if True:

    # Leave-one-out split
    training_files = annot_files.copy()
    validation_files = [annot_files[i]]
    del training_files[i]
    
    print(" *** ", validation_files, " *** ")

    # Create model
    model = mnmodel.MicronucleiModel(
        DIRECTORY, 
        device, 
        training_files=training_files, 
        validation_files=validation_files, 
        patch_size=PATCH_SIZE,
        scale_factor=SCALE_FACTOR
    )
    
    # Train
    model.train(EPOCHS, BATCH_SIZE, LR, output_dir=OUTPUT_DIR)
    
    # Validate
    model.validate()
    
    # Save
    model.save(outdir=OUTPUT_DIR)





