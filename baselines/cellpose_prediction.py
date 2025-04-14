import os
from cellpose import models
from cellpose.io import imread
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import sys
sys.path.append('../')
import skimage
import wandb

import src.dinomn.evaluation
import src.dinomn.mnds

ARCHITECTURE = "Cellpose predictions - Inference time"
CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/validation_no_rescale/'

SCALE_FACTOR = 1.0

files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
validation_files = [x for x in filelist if x.endswith('.phenotype.tif')]
validation_files.sort()

key_file = open('/scr/yren/wandb_key.txt', 'r')
key = key_file.readline()
wandb.login(key=key)

model = models.Cellpose(model_type='cyto3')
channels = [0,0]


for i in range(len(validation_files)):
    imid = validation_files[i].split('.')[0]
    
    DIAM = 15
    
    wandb.init(
        project='Best_Experiment',
        config={
            "architecture":ARCHITECTURE,
            'model':'Cellpose cyto3',
            'diameter':DIAM
        },
        name=f'{imid}',
        reinit=True
    )
    
    im = skimage.io.imread(DIRECTORY + validation_files[i])
    
    # Document inference time
    s = time.time()
    masks, flows, styles, diams = model.eval(im, diameter=DIAM, channels=channels)
    e = time.time()
    wandb.log({'Inference Time': e-s})
    
    masks = np.asarray(masks, dtype='uint16')
    
    # remove sizes > 100
    MICRON_AREA_THRESHOLD = 100
    labels = skimage.morphology.label(masks)
    micron_labels = []
    for i in range(1, len(np.unique(labels))):
        area = np.sum(labels == i)
        if area < MICRON_AREA_THRESHOLD:
            micron_labels.append(i)
            
    micro_mask = np.zeros_like(masks)
    for i in micron_labels:
        micro_mask += (labels == i)
        
    save_path = CURRENT_PATH + '/cellpose_predictions/'
    np.save(save_path + imid + '._probabilities.npy', micro_mask)
    
    # evaluation
    mn_gt = src.dinomn.mnds.read_image(DIRECTORY, imid, 'phenotype_outlines.png', scale=SCALE_FACTOR)
    src.dinomn.evaluation.segmentation_report(imid=imid, predictions=micro_mask, gt=mn_gt, intersection_ratio=0.1, report_obj='Micronuclei')

# release the resources
wandb.finish()