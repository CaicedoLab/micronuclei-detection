#!/usr/bin/env python
# coding: utf-8

import os
import time
import skimage.morphology
import torch
import skimage
import wandb
from tqdm import tqdm
import argparse
import numpy as np

import mndino.mnds as mnds
import mndino.mnmodel as mnmodel
import mndino.evaluation as evaluation



if __name__ == '__main__':

    # set CHTC writeable cahce directory for pytorch and matplotlib
    os.environ['TORCH_HOME'] = os.getcwd() + '/.cache/torch'
    torch.set_num_threads(8)
    
    parser = argparse.ArgumentParser(
        description="mnDINO Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )
    
    parser.add_argument('--path', type=str, help='mnDINO dataset path')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index.')
    parser.add_argument('--step', type=int, default=64, help='Step size of prediction box, larger value will decrease inference time but harm accuracy.')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of crops from one image that are predicted simultaneously.')
    parser.add_argument('--prob_threshold', type=float, default=0.5, help='Probability threshold to classify if each pixel is micronucleus.')
    parser.add_argument('--iou_threshold', type=float, default=0.1, help='IOU threshold, 0.1 suggested for subcellular structures.')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for aligning microscopy magnification.')
    parser.add_argument('-w', '--wandb_mode', action='store_true', help='Choose to turn on Weights and Biases')
    
    # Sample command: python3 prediction.py --path '/scr/yren/annotated_mn_datasets/' --gpu 0 --step 64 --batch_size 4 --prob_threshold 0.5 --iou_threshold 0.1 --scale 1 --wandb_mode
    
    # Fixed Hyperparameters
    PATCH_SIZE = 256
    FEATURE_SIZE = 384
    
    args = parser.parse_args()

    DIRECTORY = args.path
    VALIDATION_PATH = os.path.join(DIRECTORY, 'validation/images')
    MODEL_DIR = "model_output/"

    GPU = args.gpu
    STEP = args.step # 64 is the best
    THRESHOLD = args.prob_threshold
    IoU_THRESHOLD = args.iou_threshold # for micronuclei
    PREDICTION_BATCH = args.batch_size    
    SCALE_FACTOR = args.scale

    device = f"cuda:{GPU}" if torch.cuda.is_available() else 'cpu'
    ARCHITECTURE = f'mnDINO Inference - bicubic interpolation'


    # avoid files starting with . when untarring in CHTC
    files = os.listdir(VALIDATION_PATH)
    filelist = [file for file in files if not file.startswith('.')]
    annot_files = filelist.copy()
    annot_files.sort()

    # Validate
    predictions_dir = os.path.join(DIRECTORY, MODEL_DIR)

    # Load model and compute probabilities
    model = mnmodel.MicronucleiModel(
        device=device,
        data_dir=DIRECTORY
    )
    model_name = 'mnDINO.pth'
    model.load(os.path.join(predictions_dir, model_name))


    for i in tqdm(range(len(annot_files))):
        validation_file = annot_files[i]
        imid = validation_file.split('.')[0]
        
        config = {
                "architecture":ARCHITECTURE,
                "Loss Weight": "all default, sam ratio (0.95focal+0.05dice) + gamma=2, etc",
                "prediction_batch_size":PREDICTION_BATCH,
                "scale_factor":'Predict on images that are not scaled',
                'step':STEP,
                "feature_size":FEATURE_SIZE,
                "patch_size":PATCH_SIZE,
                "probability_threshold":THRESHOLD,
                "IoU_threshold":IoU_THRESHOLD,
                "dilation":0,
                "gaussian":'gaussian not need for prediction',
                'Number of validation images':len(annot_files)
            }
        wandb.init(
            project='mnDINO-experiment',
            config=config,
            name=f'{imid}',
            reinit=True
        )
        
        # Load image and annotations
        im = mnds.read_image(os.path.join(VALIDATION_PATH, validation_file), scale=SCALE_FACTOR)
        im = np.array((im - np.min(im))/(np.max(im) - np.min(im)), dtype="float32")
        
        mn_gt = mnds.read_image(os.path.join(VALIDATION_PATH.replace('images', 'mn_masks'), validation_file.replace('.tif', '.png')), scale=SCALE_FACTOR)
        mn_gt = mn_gt > 0 # convert to boolean (binary mask)
        
        # Document inference time
        s = time.time()
        probabilities = model.predict(im, stride=1, step=STEP, batch_size=PREDICTION_BATCH) # has model.eval() & with torch.no_grad()
        e = time.time()
        wandb.log({'Inference Time': e-s})
        filename = os.path.join(DIRECTORY, MODEL_DIR) + imid + '._probabilities'
        
        mn_pred = probabilities[0,:,:] > THRESHOLD
        labeled_mn = skimage.morphology.label(mn_pred)
        labeled_mn = np.asarray(labeled_mn, dtype='uint16') # if saving as img
        
        evaluation.segmentation_report(imid=imid, 
                                    predictions=labeled_mn, 
                                    gt=mn_gt, 
                                    intersection_ratio=IoU_THRESHOLD,
                                    wandb_mode=True)
        
        # save labeled matrices
        np.save(filename, labeled_mn)
        
    # release the resources
    torch.cuda.empty_cache()