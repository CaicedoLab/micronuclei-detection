import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import sys
sys.path.append('../')
import skimage
from skimage.measure import label
import wandb
import argparse

import mndino.evaluation as evaluation
import torch
from cellpose import io, models, train



def merge_masks(mn, nuc):
    return mn + np.where(nuc > 0, nuc + mn.max(), 0)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="Cellpose Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )
    
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index.')
    parser.add_argument('--train_path', type=str, help='mnDINO dataset path', 
                        default='/scr/data/annotated_mn_datasets/train/images')
    parser.add_argument('--save_path', type=str, help='Path to save Cellpose predictions', 
                        default='/hdd/jcaicedo/projects/micronuclei_detection/Train_and_Eval/mndino_data/baselines/cellpose_predictions')
    parser.add_argument('--finetune', action='store_true', default=False, help='specify to use frozen backbone')
    parser.add_argument('--iou_threshold', type=float, default=0.1, help='IOU threshold, 0.1 suggested for subcellular structures.')
    parser.add_argument('-w', '--wandb_mode', action='store_true', help='Choose to turn on Weights and Biases')

    args = parser.parse_args()
    GPU = args.gpu
    device = f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu'
    
    TRAIN_PATH = args.train_path
    SAVE_PATH = args.save_path
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    FINETUNE = args.finetune
    WANDB_MODE = args.wandb_mode
    
    SCALE_FACTOR = 1.0
    if FINETUNE:
        ARCHITECTURE = f"CP-SAM finetuned predictions (nuc+mn masks)"
    else:
        ARCHITECTURE = f"CP-SAM frozen predictions"
        
        
    # Load Training Images
    files = os.listdir(TRAIN_PATH)
    filelist = sorted([file for file in files if not file.startswith('.')]) # avoid files starting with . when untarring in CHTC
    
    mn_gt_files = sorted(os.listdir(TRAIN_PATH.replace('images', 'mn_masks')))
    nuc_gt_files = sorted(os.listdir(TRAIN_PATH.replace('images', 'nuclei_masks')))
    imgs = [io.imread(os.path.join(TRAIN_PATH, f)) for f in filelist]
    mn_gts = [io.imread(os.path.join(TRAIN_PATH.replace('images', 'mn_masks'), f)) for f in mn_gt_files]
    nuc_gts = [io.imread(os.path.join(TRAIN_PATH.replace('images', 'nuclei_masks'), f)) for f in nuc_gt_files]
    
    if FINETUNE:
        mn_gts = [label(gt) for gt in mn_gts]  # convert binary to instance masks for finetuning
        nuc_gts = [label(gt) for gt in nuc_gts]
        
        # concatenate
        gts = [merge_masks(m, n) for m, n in zip(mn_gts, nuc_gts)]
        
        
    # Load Test Images
    TEST_PATH = TRAIN_PATH.replace('train', 'test')
    test_files = os.listdir(TEST_PATH)
    test_filelist = [file for file in test_files if not file.startswith('.')]
    test_filelist.sort()
    
    model_type='cpsam_v2'
    model = models.CellposeModel(pretrained_model=model_type, device=torch.device(device))
    if FINETUNE:
        model_name = "cellpose_finetuned.pth"

        # default training params
        n_epochs = 100
        learning_rate = 1e-5
        weight_decay = 0.1
        batch_size = 1
        
        new_model_path, train_losses, test_losses = train.train_seg(model.net,
                                                            train_data=imgs,
                                                            train_labels=gts,
                                                            batch_size=batch_size,
                                                            n_epochs=n_epochs,
                                                            learning_rate=learning_rate,
                                                            weight_decay=weight_decay,
                                                            nimg_per_epoch=max(2, len(imgs)), # can change this,
                                                            min_train_masks=1,
                                                            model_name=model_name,
                                                            save_path=SAVE_PATH)
        
        model = models.CellposeModel(gpu=True, 
                                     pretrained_model=new_model_path, 
                                     device=torch.device(device))
    
    flow_threshold = 0.4
    cellprob_threshold = 0.0
    tile_norm_blocksize = 0
    MICRON_AREA_THRESHOLD = 300 # 300 is the best cut-off
    for i in range(len(test_filelist)):
        imid = test_filelist[i].split('.')[0]
        
        if WANDB_MODE:
            wandb.init(
                project='mnDINO-experiment',
                config={
                    "architecture": ARCHITECTURE,
                    "model": model_type,
                    'area_threshold': MICRON_AREA_THRESHOLD
                },
                name=f'{imid}',
                reinit=True,
                mode='online'
            )
        
        im = skimage.io.imread(os.path.join(TEST_PATH, test_filelist[i]))
        if SCALE_FACTOR != 1.0:
            im = skimage.transform.rescale(im, scale=SCALE_FACTOR)
            
            
        # Document inference time
        s = time.time()
        masks, flows, styles = model.eval(im, batch_size=32, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
                                  normalize={"tile_norm_blocksize": tile_norm_blocksize})
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
        gt_path = os.path.join(TEST_PATH.replace('images', 'mn_masks'), imid + '.png')
        mn_gt = skimage.io.imread(gt_path)
        if SCALE_FACTOR != 1.0:
            mn_gt = skimage.transform.rescale(mn_gt, scale=SCALE_FACTOR)
        evaluation.segmentation_report(predictions=micro_mask, gt=mn_gt, intersection_ratio=args.iou_threshold, wandb_mode=WANDB_MODE)

    # release the resources
    wandb.finish()