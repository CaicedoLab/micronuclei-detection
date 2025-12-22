#!/usr/bin/env python
# coding: utf-8

import os
import torch
import argparse
import wandb
import mndino.mnmodel as mnmodel



if __name__ == '__main__':
    # set CHTC writeable cahce directory for pytorch and matplotlib
    os.environ['TORCH_HOME'] = os.getcwd() + '/.cache/torch'
    torch.set_num_threads(8)

    parser = argparse.ArgumentParser(
        description="mnDINO Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )

    parser.add_argument('--path', type=str, help='Micronuclei dataset path')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size.')
    parser.add_argument('--loss_fn', type=str, default='combined', choices=['dice', 'focal','combined'], help='Loss function.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for the optimizer.')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor for aligning microscopy magnification.')
    parser.add_argument('--gaussian', action='store_true', default=False, help='Gaussian random size crop for model to learn distortion.')
    parser.add_argument('--edges', action='store_true', default=False, help='Recover object edges in training.')
    parser.add_argument('-w', '--wandb_mode', action='store_true', default=False, help='Choose to turn on Weights and Biases')
    
    # Sample command: python3 training_model.py --path '/hdd/jcaicedo/projects/micronuclei_detection/Train_and_Eval/mndino_data/data_to_publish/annotated_mn_datasets/' --gpu 0 --epochs 20 --loss_fn 'combined' --lr 1e-5 --scale 1.0 --gaussian --wandb_mode
    # Fixed value
    PATCH_SIZE = 256
    FEATURE_SIZE = 384
    
    args = parser.parse_args()

    DIRECTORY = args.path
    OUTPUT_DIR = "model_output/models"
    if not os.path.exists(os.path.join(DIRECTORY, OUTPUT_DIR)):
        os.makedirs(os.path.join(DIRECTORY, OUTPUT_DIR))
        
    GPU = args.gpu
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size # best training batch size
    LOSS_FN = args.loss_fn
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    SCALE_FACTOR = args.scale
    
    GAUSSIAN = args.gaussian
    EDGES = args.edges
    WANDB_MODE = args.wandb_mode

    ARCHITECTURE = 'mnDINO Training'

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
            "feature_size":FEATURE_SIZE,
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
            name=f'training',
            mode='online'
            # tags=[TAGS]
        )

    # Create model
    model = mnmodel.MicronucleiModel(
        device=device,
        data_dir=DIRECTORY,
        patch_size=PATCH_SIZE,
        scale_factor=SCALE_FACTOR,
        edges=EDGES, # False, this will recover the input edges, reducing performance
        gaussian=GAUSSIAN
    )

    # Train
    model.train(epochs=EPOCHS, 
                batch_size=BATCH_SIZE, 
                learning_rate=LR, 
                loss_fn=LOSS_FN, 
                weight_decay=WEIGHT_DECAY,
                wandb_mode=True
    )


    # Save
    model.save(outdir=OUTPUT_DIR, model_name=f'mnDINO')

    # release the resources
    torch.cuda.empty_cache()
    wandb.finish()
