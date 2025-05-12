import skimage
import numpy as np
import torch
import sys
sys.path.append('../')

from typing import Optional, Union, Tuple
from micro_sam.evaluation.model_comparison import _enhance_image
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

import dinomn.mnds
import dinomn.evaluation

import wandb
import os
from tqdm import tqdm
import time


ARCHITECTURE = "microSAM predictions - finetuned model"
CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/all_data_micronuclei_no_rescale/validation/'
SCALE_FACTOR = 1.0


def run_automatic_instance_segmentation(
    image: np.ndarray,
    checkpoint_path: Union[os.PathLike ,str],
    model_type: str = "vit_b_lm",
    device: Optional[Union[str, torch.device]] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
):
    """Automatic Instance Segmentation (AIS) by training an additional instance decoder in SAM.

    NOTE: AIS is supported only for `µsam` models.

    Args:
        image: The input image.
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
        is_tiled=(tile_shape is not None),  # whether to run automatic segmentation.
    )

    # Step 2: Get the instance segmentation for the given image.
    prediction = automatic_instance_segmentation(
        predictor=predictor,  # the predictor for the Segment Anything model.
        segmenter=segmenter,  # the segmenter class responsible for generating predictions.
        input_path=image,  # the filepath to image or the input array for automatic segmentation.
        ndim=2,  # the number of input dimensions.
        tile_shape=tile_shape,  # the tile shape for tiling-based prediction.
        halo=halo,  # the overlap shape for tiling-based prediction.
    )

    return prediction


files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
validation_files = [x for x in filelist if x.endswith('.phenotype.tif')]
validation_files.sort()

key_file = open('/scr/yren/wandb_key.txt', 'r')
key = key_file.readline()
wandb.login(key=key)

# Hyperparameters to load fine-tuned model
checkpoint_name = "sam_hela"
best_checkpoint = os.path.join(os.getcwd(), 'data_for_finetuning', "models", "checkpoints", checkpoint_name, "best.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"  # the device/GPU used for training
model_type = "vit_b_lm"


for i in range(len(validation_files)):
    imid = validation_files[i].split('.')[0]
    
    # Initialize WanDB
    wandb.init(
        project='Best_Experiment',
        config={
            "architecture":ARCHITECTURE,
            'model':f'{model_type}',
        },
        name=f'{imid}',
        reinit=True
    )
    
    im = skimage.io.imread(DIRECTORY + validation_files[i])
    
    # Document Inference Time
    s = time.time()
    prediction = run_automatic_instance_segmentation(
        image=im,
        checkpoint_path=best_checkpoint,
        model_type=model_type,
        device=device
    )
    e = time.time()
    wandb.log({'Inference Time': e-s})
    
    # filter sizes > 100 to get micronuclei prediction
    MICRON_AREA_THRESHOLD = 100
    wandb.log({'micronuclei area threshold':MICRON_AREA_THRESHOLD})
    labels = skimage.morphology.label(prediction)
    micron_labels = []
    for i in range(1, len(np.unique(labels))):
        area = np.sum(labels == i)
        if area < MICRON_AREA_THRESHOLD:
            micron_labels.append(i)
            
    micro_mask = np.zeros_like(prediction)
    for i in micron_labels:
        micro_mask += (labels == i)
    
    save_path = CURRENT_PATH + '/microSAM_predictions/'
    np.save(save_path + imid + '._probabilities.npy', micro_mask)
    
    # evaluation
    mn_gt = dinomn.mnds.read_image(DIRECTORY, imid, 'phenotype_outlines.png', scale=SCALE_FACTOR)
    dinomn.evaluation.segmentation_report(imid=imid, predictions=micro_mask, gt=mn_gt, intersection_ratio=0.1, report_obj='Micronuclei')

# release the resources
wandb.finish()