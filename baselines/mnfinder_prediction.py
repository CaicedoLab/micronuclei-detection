# Parallelize on CHTC to get predicted masks
import os
import sys
sys.path.append('../')
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
from tifffile import tifffile
from matplotlib import pyplot as plt
from PIL import Image
import skimage
from skimage.measure import label
from skimage.color import label2rgb
import wandb
# import cv2

from skimage.exposure import rescale_intensity
from skimage.segmentation import clear_border

from mnfinder import MNClassifier
import dinomn.evaluation as evaluation
import dinomn.mnds as mnds

ARCHITECTURE = "MNFinder predictions - Inference time"
CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/all_data_micronuclei_no_rescale/validation/'


files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
validation_files = [x for x in filelist if x.endswith('.phenotype.tif')]
validation_files.sort()


key_file = open('/scr/yren/wandb_key.txt', 'r')
key = key_file.readline()
wandb.login(key=key)
    
attention_model = MNClassifier.get_model('Attention')

for i in range(len(validation_files)):
    imid = validation_files[i].split('.')[0]
    
    SCALE_FACTOR = 1.0
   
    wandb.init(
        project='Best_Experiment',
        config={
            "architecture":ARCHITECTURE,
            'model':'MNFinder',
            'scale_factor':SCALE_FACTOR
        },
        name=f'{imid}',
        reinit=True
    )

    im = mnds.read_image(DIRECTORY, imid, 'phenotype.tif', scale=SCALE_FACTOR)
    
    # Document inference time
    s = time.time()
    labels = attention_model.predict(im)
    e = time.time()
    wandb.log({'Inference Time': e-s})
    
    micro_labels = labels[:,:,2]

    save_path = CURRENT_PATH + '/mnfinder_predictions/'
    np.save(save_path + imid + '._probabilities.npy', micro_labels)

    # evaluation
    mn_gt = mnds.read_image(DIRECTORY, imid, 'phenotype_outlines.png', scale=SCALE_FACTOR)
    evaluation.segmentation_report(imid=imid, predictions=micro_labels, gt=mn_gt, intersection_ratio=0.1, report_obj='Micronuclei')

# release the resources
wandb.finish()