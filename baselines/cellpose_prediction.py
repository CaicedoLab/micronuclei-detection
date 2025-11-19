import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import sys
sys.path.append('../')
import skimage
import wandb
import argparse

import mndino.evaluation as evaluation
import torch
from cellpose import io, models, train


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="Cellpose Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )
    
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index.')
    parser.add_argument('--train_path', type=str, help='mnDINO dataset path') # '/scr/yren/annotated_mn_datasets/test/images/'
    parser.add_argument('--save_path', type=str, help='Path to save Cellpose predictions')
    parser.add_argument('--frozen', action='store_true', help='specify to use frozen backbone')
    parser.add_argument('-w', '--wandb_mode', action='store_true', help='Choose to turn on Weights and Biases')

    args = parser.parse_args()
    GPU = args.gpu
    device = f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu'
    
    PATH = args.train_path
    SAVE_PATH = args.save_path
    FROZEN = args.frozen
    WANDB_MODE = args.wandb_mode
    
    SCALE_FACTOR = 1.0
    ARCHITECTURE = f"Cellpose finetuned predictions"

    # Train
    TRAIN_PATH = PATH.replace('test', 'train')
    files = os.listdir(TRAIN_PATH)
    filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
    filelist.sort()

    test_files = os.listdir(PATH)
    test_filelist = [file for file in test_files if not file.startswith('.')]
    
    gts_files = os.listdir(TRAIN_PATH.replace('images', 'mn_masks'))
    
    imgs = [io.imread(os.path.join(TRAIN_PATH, f)) for f in filelist]
    gts = [io.imread(os.path.join(TRAIN_PATH.replace('images', 'mn_masks'), f)) for f in gts_files]

    model = models.CellposeModel(gpu=True, model_type='cyto3', device=torch.device(device))
    channels = [0,0] # grayscale in cellpose 3
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=imgs,
        train_labels=gts,
        channels=channels,              # Use first channel (grayscale)
        channel_axis=None,            # No channel dimension in your data
        weight_decay=1e-6,
        learning_rate=1e-5,
        n_epochs=100,
        normalize=True,
        min_train_masks=1,
        model_name=f"cellpose_finetuned.pth",
        save_path=SAVE_PATH
    )


    MICRON_AREA_THRESHOLD = 300 # 300 is the best cut-off
    DIAMETER = model.diam_labels.copy()
    for i in range(len(test_filelist)):
        imid = test_filelist[i].split('.')[0]
        
        if WANDB_MODE:
            wandb.init(
                project='mnDINO-experiment',
                config={
                    "architecture":ARCHITECTURE,
                    'model':'Cellpose finetuned',
                    'diameter':DIAMETER,
                    'area_threshold':MICRON_AREA_THRESHOLD
                },
                name=f'{imid}',
                reinit=True,
                mode='online'
            )
        
        im = skimage.io.imread(os.path.join(PATH, test_filelist[i]))
        if SCALE_FACTOR != 1.0:
            im = skimage.transform.rescale(im, scale=SCALE_FACTOR)
        
        # Document inference time
        s = time.time()
        masks = model.eval(im, channels=channels, diameter=DIAMETER)[0]
        e = time.time()
        if WANDB_MODE:
            wandb.log({'Inference Time': e-s})
        print(f'{imid}, Inference time used: {e - s: .2f}')
        
        MASKS = np.asarray(masks, dtype='uint16')
        
        micron_labels = []
        for i in range(1, len(np.unique(MASKS))):
            area = np.sum(MASKS == i)
            if area < MICRON_AREA_THRESHOLD:
                micron_labels.append(i)
                
        micro_mask = np.zeros_like(masks)
        for i in micron_labels:
            micro_mask += (MASKS == i)
            
        np.save(os.path.join(SAVE_PATH, imid + '._probabilities.npy'), micro_mask)
        
        # evaluation
        gt_path = os.path.join(PATH.replace('images', 'mn_masks'), imid + '.png')
        mn_gt = skimage.io.imread(gt_path)
        if SCALE_FACTOR != 1.0:
            mn_gt = skimage.transform.rescale(mn_gt, scale=SCALE_FACTOR)
        evaluation.segmentation_report(imid=imid, predictions=micro_mask, gt=mn_gt, intersection_ratio=0.1, wandb_mode=WANDB_MODE)

    # release the resources
    wandb.finish()