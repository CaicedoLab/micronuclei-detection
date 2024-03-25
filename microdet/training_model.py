#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import torch
import wandb
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
OUTPUT_DIR = "/model_output/nuclei_experiments/"

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

# Initiate Weights and Biases Configuration
# wandb.login(key='b3f4f9254c123781af918799b27affa92d8f4eeb')
# wandb.init(
#     project='micronuclei-segmentation-training',
    
#     # hyperparameters
#     config={
#         "architecture":"3 blocks, 1 2x2 upscale, 2 3x3 conv layers",
#         "learning_rate":LR,
#         "epochs": EPOCHS,
#         "feature_size":FEATURE_SIZE,
#         "batch_size":BATCH_SIZE,
#         "patch_size":PATCH_SIZE,
#         "fine_tuning":False,
#         "Scheduler":"Cos scheduler",
#         "Loss": "Weighted Dice Loss"
#     }
# )

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
        scale_factor=SCALE_FACTOR,
        edges=True
    )
    
    # Train
    model.train(EPOCHS, BATCH_SIZE, LR, loss_fn='dice', output_dir=OUTPUT_DIR, finetune=False)
    
    # Validate
    model.validate()
    
    # Save
    model.save(outdir=OUTPUT_DIR)


# wandb.finish()