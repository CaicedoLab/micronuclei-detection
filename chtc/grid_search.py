#!/usr/bin/env python
# coding: utf-8

import os
import torch
import argparse
import time
import skimage.morphology
import torch
import skimage
import wandb
from tqdm import tqdm
import argparse
import numpy as np

import sys
sys.path.append('../')
import mndino.mnmodel as mnmodel
import mndino.mnds as mnds
import mndino.mnmodel as mnmodel
import mndino.evaluation as evaluation



if __name__ == '__main__':
    # set CHTC writeable cahce directory for pytorch and matplotlib
    os.environ['TORCH_HOME'] = os.getcwd() + '/.cache/torch'
    torch.set_num_threads(8)

    parser = argparse.ArgumentParser(
        description="mnDINO Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )

    # Train
    parser.add_argument('--id', type=int)
    parser.add_argument('--path', type=str, help='Micronuclei dataset path', default='/scratch/yren86/annotated_mn_datasets')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index.')
    parser.add_argument('--model_type', default='dino', choices=['dino', 'unet'], help='Choose model type over DINO-based architecture or simple 2D Unet.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size.')
    parser.add_argument('--loss_fn', type=str, default='combined', choices=['dice', 'focal','combined'], help='Loss function.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for the optimizer.')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for aligning microscopy magnification.')
    parser.add_argument('--gaussian', action='store_true', default=False, help='Gaussian random size crop for model to learn distortion.')
    parser.add_argument('--edges', action='store_true', default=False, help='Recover object edges in training.')
    
    # Prediction
    parser.add_argument('--test_set', action='store_true', default=False, help='Turn on to choose test set, otherwise validation set')
    parser.add_argument('--step', type=int, default=32, help='Step size of prediction box, larger value will decrease inference time but harm accuracy.')
    parser.add_argument('--prob_threshold', type=float, default=0.5, help='Probability threshold to classify if each pixel is micronucleus.')
    parser.add_argument('--iou_threshold', type=float, default=0.1, help='IOU threshold, 0.1 suggested for subcellular structures.')
    parser.add_argument('-w', '--wandb_mode', action='store_true', default=False, help='Choose to turn on Weights and Biases')
    
    
    PATCH_SIZE = 256

    
    args = parser.parse_args()
    # TAGS = args.tags
    # INDEX = args.iteration

    DIRECTORY = args.path
    OUTPUT_DIR = "model_output/models"
    if not os.path.exists(os.path.join(DIRECTORY, OUTPUT_DIR)):
        os.makedirs(os.path.join(DIRECTORY, OUTPUT_DIR))
        
    GPU = args.gpu
    MODEL_TYPE = args.model_type
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size # best training batch size
    LOSS_FN = args.loss_fn
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    SCALE_FACTOR = args.scale
    
    GAUSSIAN = args.gaussian
    EDGES = args.edges
    WANDB_MODE = args.wandb_mode

    ARCHITECTURE = f'mnDINO (dinov3) Training ({args.id})'

    device = f"cuda:{GPU}" if torch.cuda.is_available() else 'cpu'
    

    if WANDB_MODE:
        num_training_files = len(os.listdir(os.path.join(DIRECTORY, 'train/images')))
        num_validation_files = len(os.listdir(os.path.join(DIRECTORY, 'validation/images')))
        config = {
            "architecture":ARCHITECTURE,
            "Loss": LOSS_FN,
            "Loss Weight": "all default, sam ratio (0.95focal+0.05dice) + gamma=2, etc", 
            "training_batch_size":BATCH_SIZE,
            "start_learning_rate":LR,
            "lr_scheduler":"Cosine",
            "scale_factor":'Trained on non-scaled images',
            "epochs": EPOCHS,
            "patch_size":PATCH_SIZE,
            "weight_decay":WEIGHT_DECAY,
            "gaussian":GAUSSIAN,
            'edges':EDGES,
            'Number of training images':num_training_files,
            'Number of validation images':num_validation_files
        }
        wandb.init(
            project='mnDINO-experiment',
            config=config,
            name=f'training_{args.id}',
            mode='online',
            tags=['dinov3-chtc']
        )

    # Create model
    model = mnmodel.MicronucleiModel(
        device=device,
        data_dir=DIRECTORY,
        patch_size=PATCH_SIZE,
        scale_factor=SCALE_FACTOR,
        edges=EDGES, # False, this will recover the input edges, reducing performance
        gaussian=GAUSSIAN,
        model_type=MODEL_TYPE
    )

    # Train
    model.train(epochs=EPOCHS, 
                batch_size=BATCH_SIZE, 
                learning_rate=LR, 
                loss_fn=LOSS_FN, 
                weight_decay=WEIGHT_DECAY,
                wandb_mode=WANDB_MODE
    )


    # Save
    model.save(outdir=OUTPUT_DIR, model_name=f'mnDINO_{args.id}')

    
    
    # Prediction
    # avoid files starting with . when untarring in CHTC
    DIRECTORY = args.path
    MODEL_DIR = "model_output/models"
    PRED_DIR = 'model_output/'
    THRESHOLD = args.prob_threshold
    IoU_THRESHOLD = args.iou_threshold
    
    if_test = args.test_set
    if if_test:
        which_set = 'test'
        PATH = os.path.join(DIRECTORY, 'test/images')
    else:
        which_set = 'validation'
        PATH = os.path.join(DIRECTORY, 'validation/images')
    
    ARCHITECTURE = f'mnDINO Inference (dinov3) - {which_set} set ({args.id})'
        
    files = os.listdir(PATH)
    filelist = [file for file in files if not file.startswith('.')]
    annot_files = sorted(filelist.copy())

    # Validate
    models_dir = os.path.join(DIRECTORY, MODEL_DIR)
    predictions_dir = os.path.join(DIRECTORY, PRED_DIR)

    # Load model and compute probabilities
    model = mnmodel.MicronucleiModel(
        device=device,
        data_dir=DIRECTORY,
        model_type=MODEL_TYPE
    )
    model_name = f'mnDINO_{args.id}.pth'
    # model_name = f'mnDINO_{INDEX}.pth'
    model.load(model_path=os.path.join(models_dir, model_name))


    for i in tqdm(range(len(annot_files))):
        validation_file = annot_files[i]
        imid = validation_file.split('.')[0]
        
        if WANDB_MODE:
            config = {
                    "architecture":ARCHITECTURE,
                    "Loss Weight": "all default, sam ratio (0.95focal+0.05dice) + gamma=2, etc",
                    "prediction_batch_size":BATCH_SIZE,
                    "scale_factor":'Predict on images that are not scaled',
                    'step':args.step,
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
                reinit=True,
                mode='online',
                # tags=[TAGS]
            )
        
        # Load image and annotations
        im = mnds.read_image(os.path.join(PATH, validation_file), scale=SCALE_FACTOR)
        im = np.array((im - np.min(im))/(np.max(im) - np.min(im)), dtype="float32")
        
        mn_gt = mnds.read_image(os.path.join(PATH.replace('images', 'mn_masks'), validation_file.replace('.tif', '.png')), scale=SCALE_FACTOR)
        mn_gt = mn_gt > 0 # convert to boolean (binary mask)
        
        # Document inference time
        s = time.time()
        probabilities = model.predict(im, stride=1, step=args.step, batch_size=args.batch_size) # has model.eval() & with torch.no_grad()
        e = time.time()
        if WANDB_MODE:
            wandb.log({'Inference Time': e-s})
        print(f'{imid}, Inference time used: {e - s: .2f}')
        filename = os.path.join(DIRECTORY, PRED_DIR) + imid + '._probabilities'
        
        mn_pred = probabilities[0,:,:] > THRESHOLD
        labeled_mn = skimage.morphology.label(mn_pred)
        labeled_mn = np.asarray(labeled_mn, dtype='uint16') # if saving as img
        
        precision, recall = evaluation.segmentation_report(predictions=labeled_mn, 
                                                           gt=mn_gt, 
                                                           intersection_ratio=IoU_THRESHOLD,
                                                           wandb_mode=WANDB_MODE)
        
        print(f'{imid}, Precision: {precision}, Recall: {recall}')
        # save labeled matrices
        np.save(filename, labeled_mn)
        
    # release the resources
    torch.cuda.empty_cache()