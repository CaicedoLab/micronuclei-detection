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

ARCHITECTURE = "MNFinder predictions on 47 validation images aligned to mnfinder scales"
CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/validation_no_rescale/'


files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
validation_files = [x for x in filelist if x.endswith('.phenotype.tif')]
validation_files.sort()


key_file = open('../microdet/wandb_key.txt', 'r')
key = key_file.readline()
wandb.login(key=key)
    
attention_model = MNClassifier.get_model('Attention')

df = pd.read_csv('/scr/yren/all_data_micronuclei/metadata.csv')
data_mnfinder = df[df.datasets=='mnfinder_validation']
for i in range(len(validation_files)):
    imid = validation_files[i].split('.')[0]
    
    # SCALE_FACTOR = 1.0
    mag = df.loc[df.filenames == validation_files[i], 'magnification'].iloc[0]
    micron = df.loc[df.filenames == validation_files[i], 'micron'].iloc[0]
    SCALE_FACTOR = round((20/11/(mag/micron)), 1)
   
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

    im = microdet.mnds.read_image(DIRECTORY, imid, 'phenotype.tif', scale=SCALE_FACTOR)
    
    labels = attention_model.predict(im)
    micro_labels = labels[:,:,2]

    save_path = DIRECTORY + 'predictions_scale_aligned_to_mnfinder/'
    np.save(save_path + imid + '._probabilities.npy', micro_labels)

    # evaluation
    mn_gt = microdet.mnds.read_image(DIRECTORY, imid, 'phenotype_outlines.png', scale=SCALE_FACTOR)
    microdet.evaluation.segmentation_report(imid=imid, predictions=micro_labels, gt=mn_gt, intersection_ratio=0.1, report_obj='Micronuclei')

# release the resources
wandb.finish()