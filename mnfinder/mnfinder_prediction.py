# Parallelize on CHTC to get predicted masks
import os
import sys
sys.path.append('../')
from pathlib import Path
from tqdm import tqdm
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
import microdet.evaluation
import microdet.mnds

ARCHITECTURE = "MNFinder predictions on 47 validation images"
CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/validation/'

SCALE_FACTOR = 1.0


files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
validation_files = [x for x in filelist if x.endswith('.phenotype.tif')]
validation_files.sort()

# if len(sys.argv) < 2:
#     print("Use: python mnfinder_prediction.py imid")
#     sys.exit()

# i = int(sys.argv[1])


key_file = open('../microdet/wandb_key.txt', 'r')
key = key_file.readline()
wandb.login(key=key)
    
attention_model = MNClassifier.get_model('Attention')

for i in range(len(validation_files)):
    imid = validation_files[i].split('.')[0]
   
    wandb.init(
        project='Best_Experiment',
        config={
            "architecture":ARCHITECTURE,
            'model':'MNFinder'
        },
        name=f'{imid}',
        reinit=True
    )

    im = skimage.io.imread(DIRECTORY + validation_files[i])
    
    labels = attention_model.predict(im)
    micro_labels = labels[:,:,2]
    micro_labels = np.asarray(micro_labels, dtype='uint16')

    save_path = DIRECTORY + 'mnfinder_predictions_output/'
    np.save(save_path + imid + '._probabilities.npy', micro_labels)

    # evaluation
    mn_gt = microdet.mnds.read_image(DIRECTORY, imid, 'phenotype_outlines.png', scale=SCALE_FACTOR)
    microdet.evaluation.segmentation_report(imid=imid, predictions=micro_labels, gt=mn_gt, intersection_ratio=0.1, report_obj='Micronuclei')

# release the resources
wandb.finish()