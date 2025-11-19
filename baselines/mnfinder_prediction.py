# Parallelize on CHTC to get predicted masks
import os
import sys
sys.path.append('../')
from tqdm import tqdm
import argparse
import time
import numpy as np
# import pandas as pd
# from tifffile import tifffile
import skimage
import wandb

from skimage.exposure import rescale_intensity
from skimage.segmentation import clear_border

from mnfinder import MNClassifier
import mndino.evaluation as evaluation
# import mndino.mnds as mnds


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="MNFinder Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )
    
    parser.add_argument('--train_path', type=str, help='mnDINO dataset path') # '/scr/yren/annotated_mn_datasets/test/images/'
    parser.add_argument('--save_path', type=str, help='Path to save MNFinder predictions')
    parser.add_argument('-w', '--wandb_mode', action='store_true', help='Choose to turn on Weights and Biases')
    
    args = parser.parse_args()
    PATH = args.train_path
    SAVE_PATH = args.save_path
    WANDB_MODE = args.wandb_mode

    files = os.listdir(PATH)
    filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
    # validation_files = [x for x in filelist if x.endswith('.phenotype.tif')]
    filelist.sort()
        
    combined_model = MNClassifier.get_model() # default: combined

    ARCHITECTURE = "MNFinder predictions"
    SCALE_FACTOR = 1.0
    
    for i in tqdm(range(len(filelist))):
        imid = filelist[i].split('.')[0]

        if WANDB_MODE:
            wandb.init(
                project='mnDINO-experiment',
                config={
                    "architecture":ARCHITECTURE,
                    'model':'MNFinder',
                    'scale_factor':SCALE_FACTOR
                },
                name=f'{imid}',
                reinit=True,
                mode='online'
            )

        im_path = os.path.join(PATH, imid + '.tif')
        im = skimage.io.imread(im_path)
        if SCALE_FACTOR != 1.0:
            im = skimage.transform.rescale(im, scale=SCALE_FACTOR)
        
        # Document inference time
        s = time.time()
        labels = combined_model.predict(im)
        e = time.time()
        if WANDB_MODE:
            wandb.log({'Inference Time': e-s})
        print(f'{imid}, Inference time used: {e - s: .2f}')
        micro_labels = labels[:,:,1]

        # save_path = CURRENT_PATH + '/mnfinder_predictions/'
        
        np.save(os.path.join(SAVE_PATH, imid + '._probabilities.npy'), micro_labels)

        # evaluation
        # mn_gt = mnds.read_image(PATH.replace('images', 'mn_masks'), imid, 'phenotype_outlines.png', scale=SCALE_FACTOR)
        # MNFinder tensorflow conflicts with torch, do not import mnds file.
        gt_path = os.path.join(PATH.replace('images', 'mn_masks'), imid + '.png')
        mn_gt = skimage.io.imread(gt_path)
        if SCALE_FACTOR != 1.0:
            mn_gt = skimage.transform.rescale(mn_gt, scale=SCALE_FACTOR)
        evaluation.segmentation_report(imid=imid, predictions=micro_labels, gt=mn_gt, intersection_ratio=0.1, wandb_mode=WANDB_MODE)

    # release the resources
    wandb.finish()