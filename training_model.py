#!/usr/bin/env python
# coding: utf-8

import os
import torch
import argparse
import wandb
import dinomn.mnmodel as mnmodel



if __name__ == '__main__':
    
    # CURRENT_PATH = os.getcwd()
    # DIRECTORY = CURRENT_PATH + '/annotated_mn_datasets/'
    # OUTPUT_DIR = "model_output/"
    # if not os.path.exists(DIRECTORY + OUTPUT_DIR):
    #     os.makedirs(DIRECTORY + OUTPUT_DIR)

    # set CHTC writeable cahce directory for pytorch and matplotlib
    os.environ['TORCH_HOME'] = os.getcwd() + '/.cache/torch'
    torch.set_num_threads(8)

    parser = argparse.ArgumentParser(
        description="DinoMN Training",
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

    parser.add_argument('--finetune', action='store_true', help='Finetuning the DINOv2 backbone.')
    parser.add_argument('--gaussian', action='store_true', help='Gaussian random size crop for model to learn distortion.')
    parser.add_argument('--edges', action='store_true', help='Recover object edges in training.')
    parser.add_argument('-w', '--wandb_mode', action='store_true', help='Choose to turn on Weights and Biases')
    parser.set_defaults(finetune=True, gaussian=True, edges=False, wandb_mode=False)

    # Sample command: python3 training_model.py --path '/scr/yren/annotated_mn_datasets/' --gpu 0 --epochs 20 --loss_fn 'combined' --lr 1e-6 --scale 1.0 --finetune --gaussian --wandb_mode
    
    # Fixed value for DINOv2
    PATCH_SIZE = 256
    FEATURE_SIZE = 384
    
    args = parser.parse_args()

    DIRECTORY = args.path
    OUTPUT_DIR = "model_output/"
    if not os.path.exists(DIRECTORY + OUTPUT_DIR):
        os.makedirs(DIRECTORY + OUTPUT_DIR)
        
    GPU = args.gpu
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size # best training batch size
    LOSS_FN = args.loss_fn
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    SCALE_FACTOR = args.scale
    
    FINETUNE = args.finetune
    GAUSSIAN = args.gaussian
    EDGES = args.edges
    
    if args.wandb_mode:
        WANDB_MODE = 'online'
    else:
        WANDB_MODE = 'disabled'

    ARCHITECTURE = 'mnDINO Training'

    device = f"cuda:{GPU}" if torch.cuda.is_available() else 'cpu'
    
    num_training_files = len(os.listdir(DIRECTORY + 'train/images'))
    num_validation_files = len(os.listdir(DIRECTORY + 'validation/images'))


    config = {
        "architecture":ARCHITECTURE,
        "Loss": LOSS_FN,
        "Loss Weight": "all default, sam ratio (0.95focal+0.05dice) + gamma=2, etc",
        "fine_tuning":FINETUNE,
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
        project='Best_Experiment',
        config=config,
        name=f'model version 3 (test new predict())',
        mode=WANDB_MODE
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
                finetune=FINETUNE,
                weight_decay=WEIGHT_DECAY,
                wandb_mode=True
    )


    # Save
    model.save(outdir=OUTPUT_DIR, model_name='DinoMN')

    # release the resources
    torch.cuda.empty_cache()
    wandb.finish()
