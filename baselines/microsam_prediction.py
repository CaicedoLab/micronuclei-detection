import skimage
import numpy as np
import torch
import sys
sys.path.append('../')

import warnings
warnings.filterwarnings("ignore")

import os
from typing import Union, Tuple, Optional
import torch
from torch_em.data import MinInstanceSampler
import micro_sam.training as sam_training
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

# import dinomn.mnds as mnds
import mndino.evaluation as evaluation

import wandb
import time
import argparse
from tqdm import tqdm


if __name__ == '__main__':
    ###
    ### Need to preprocess the images into 256x256, just care the location that has micronuclei.
    ###
    parser = argparse.ArgumentParser(
        description="microSAM Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )
    
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index.')
    parser.add_argument('--train_path', type=str, help='mnDINO dataset path', default='/scr/yren/microsam_data/test/')
    parser.add_argument('--pred_path', type=str, help='folder of images that microSAM predicts', default='/scr/yren/annotated_mn_datasets/test/images/')
    parser.add_argument('--save_path', type=str, help='Path to save microSAM predictions', default='/scr/yren/microsam_data/microsam_predictions/')
    parser.add_argument('--frozen', action='store_true', help='specify to use frozen backbone')
    parser.add_argument('-w', '--wandb_mode', action='store_true', help='Choose to turn on Weights and Biases')

    args = parser.parse_args()
    
    GPU = args.gpu
    torch.cuda.set_device(GPU)
    device = f"cuda" if torch.cuda.is_available() else "cpu"  # the device/GPU used for training
    
    PATH = args.train_path
    PRED_PATH = args.pred_path
    SAVE_PATH = args.save_path
    if args.wandb_mode:
        WANDB_MODE = 'online'
    else:
        WANDB_MODE = 'disabled'
    FROZEN = args.frozen
    SCALE_FACTOR = 1.0

    if FROZEN:
        ARCHITECTURE = f"microSAM predictions - frozen"
    else:
        ARCHITECTURE = f"microSAM predictions - finetuned"

    
    def run_automatic_instance_segmentation(
        image: np.ndarray,
        ndim: int,
        checkpoint_path: Optional[Union[os.PathLike, str]] = None,
        model_type: str = "vit_b_lm",
        device: Optional[Union[str, torch.device]] = None,
        tile_shape: Optional[Tuple[int, int]] = None,
        halo: Optional[Tuple[int, int]] = None,
    ):
        """Automatic Instance Segmentation (AIS) by training an additional instance decoder in SAM.

        NOTE: AIS is supported only for `µsam` models.

        Args:
            image: The input image.
            ndim: The number of dimensions for the input data.
            checkpoint_path: The path to stored checkpoints.
            model_type: The choice of the `µsam` model.
            device: The device to run the model inference.
            tile_shape: The tile shape for tiling-based segmentation.
            halo: The overlap shape on each side per tile for stitching the segmented tiles.

        Returns:
            The instance segmentation.
        """
        # Step 1: Get the 'predictor' and 'segmenter' to perform automatic instance segmentation.
        predictor, segmenter = get_predictor_and_segmenter(
            model_type=model_type,  # choice of the Segment Anything model
            checkpoint=checkpoint_path,  # overwrite to pass your own finetuned model.
            device=device,  # the device to run the model inference.
            amg=False,  # set the automatic segmentation mode to AIS.
            is_tiled=(tile_shape is not None),  # whether to run automatic segmentation with tiling.
        )

        # Step 2: Get the instance segmentation for the given image.
        prediction = automatic_instance_segmentation(
            predictor=predictor,  # the predictor for the Segment Anything model.
            segmenter=segmenter,  # the segmenter class responsible for generating predictions.
            input_path=image,  # the filepath to image or the input array for automatic segmentation.
            ndim=ndim,  # the number of input dimensions.
            tile_shape=tile_shape,  # the tile shape for tiling-based prediction.
            halo=halo,  # the overlap shape for tiling-based prediction.
        )

        return prediction

    # FROZEN = TRUE
    if FROZEN:
        FINETUNE = False
    else:
        FINETUNE = True
        
    if FINETUNE:
        # Data Loader
        raw_key, label_key = '*.tif', '*.png'

        train_dir = PATH.replace('test', 'train')
        val_dir = PATH.replace('test', 'validation')
        train_instance_segmentation = True
        train_segmentation_dir = train_dir
        val_segmentation_dir = val_dir

        batch_size = 1  # the training batch size
        patch_shape = (256, 256)  # the size of patches for training
        sampler = MinInstanceSampler(min_size=25)

        train_loader = sam_training.default_sam_loader(
            raw_paths=train_dir,
            raw_key=raw_key,
            label_paths=train_segmentation_dir,
            label_key=label_key,
            with_segmentation_decoder=train_instance_segmentation,
            patch_shape=patch_shape,
            batch_size=batch_size,
            is_seg_dataset=True,
            rois=None,
            shuffle=True,
            raw_transform=sam_training.identity,
            sampler=sampler
        )

        val_loader = sam_training.default_sam_loader(
            raw_paths=val_dir,
            raw_key=raw_key,
            label_paths=val_segmentation_dir,
            label_key=label_key,
            with_segmentation_decoder=train_instance_segmentation,
            patch_shape=patch_shape,
            batch_size=batch_size,
            is_seg_dataset=True,
            rois=None,
            shuffle=True,
            raw_transform=sam_training.identity,
            sampler=sampler
        )


        # Finetuning
        n_objects_per_batch = 5  # the number of objects per batch that will be sampled
        n_epochs = 5
        model_type = "vit_b_lm"
        checkpoint_name = "sam_finetuned_all"
        
        CKPT_PATH = PATH.replace('test/', '')
        best_checkpoint = os.path.join(CKPT_PATH, 'microsam_predictions', 'models', 'checkpoints', checkpoint_name, 'best.pt')

        # if best_checkpoint exists, skip training
        if not os.path.exists(best_checkpoint):
            sam_training.train_sam(
                name=checkpoint_name,
                save_root=os.path.join(CKPT_PATH, "microsam_predictions/models"),
                model_type=model_type,
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=n_epochs,
                n_objects_per_batch=n_objects_per_batch,
                with_segmentation_decoder=train_instance_segmentation,
                device=device
            )

    # Test
    files = os.listdir(PRED_PATH) # test path
    filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
    test_files = [x for x in filelist if x.endswith('.tif')]
    test_files.sort()

    # Load the best checkpoint
    MICRON_AREA_THRESHOLD = 300
    for i in tqdm(range(len(test_files))):
        imid = test_files[i].split('.')[0]
        
        # Initialize WanDB
        wandb.init(
            project='mnDINO-experiment',
            config={
                "architecture":ARCHITECTURE,
                'model':f'microSAM finetuned',
                'area_threshold':MICRON_AREA_THRESHOLD
            },
            name=f'{imid}',
            reinit=True,
            mode=WANDB_MODE
        )
        
        im = skimage.io.imread(PRED_PATH + test_files[i])
        H,W = im.shape
        
        # Document Inference Time
        if FROZEN:
            s = time.time()
            model_choice = 'vit_b_lm'
            if (H > 1024) and (W > 1024):
                torch.cuda
                prediction = run_automatic_instance_segmentation(im, ndim=2, model_type=model_choice, device=device, tile_shape=(1024, 1024), halo=(256, 256))
            else:
                prediction = run_automatic_instance_segmentation(im, ndim=2, model_type=model_choice, device=device)
            e = time.time()
            if WANDB_MODE:
                wandb.log({'Inference Time': e-s})
        else: # finetune
            s = time.time()
            if (H > 1024) and (W > 1024):
                prediction = run_automatic_instance_segmentation(
                    image=im,
                    ndim=2,
                    checkpoint_path=best_checkpoint,
                    model_type=model_type,
                    device=device,
                    tile_shape=(1024, 1024), 
                    halo=(256, 256)
                )
            else:
                prediction = run_automatic_instance_segmentation(
                    image=im,
                    ndim=2,
                    checkpoint_path=best_checkpoint,
                    model_type=model_type,
                    device=device
                )
            e = time.time()
            if WANDB_MODE:
                wandb.log({'Inference Time': e-s})
        
        print(f'{imid}, Inference time used: {e - s: .2f}')
        prediction = np.asarray(prediction, dtype='uint16')

        labels = skimage.morphology.label(prediction)
        micron_labels = []
        for i in range(1, len(np.unique(labels))):
            area = np.sum(labels == i)
            if area < MICRON_AREA_THRESHOLD:
                micron_labels.append(i)
                
        micro_mask = np.zeros_like(prediction)
        for i in micron_labels:
            micro_mask += (labels == i)
        
        np.save(os.path.join(SAVE_PATH, imid + f'._probabilities.npy'), micro_mask)
        
        # evaluation
        gt_path = os.path.join(PRED_PATH.replace('images', 'mn_masks'), imid + f'.png')
        mn_gt = skimage.io.imread(gt_path)
        evaluation.segmentation_report(imid=imid, predictions=micro_mask, gt=mn_gt, intersection_ratio=0.1, wandb_mode=WANDB_MODE)

    # release the resources
    wandb.finish()