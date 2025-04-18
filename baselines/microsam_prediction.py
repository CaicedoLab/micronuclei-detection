import skimage
import numpy as np
import torch
import sys
sys.path.append('../')

from typing import Optional, Union, Tuple
from micro_sam.evaluation.model_comparison import _enhance_image
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

import src.dinomn.mnds
import src.dinomn.evaluation

import wandb
import os
from tqdm import tqdm
import time


ARCHITECTURE = "microSAM predictions"
CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/all_data_micronuclei_no_rescale/validation/'
SCALE_FACTOR = 1.0


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


files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
validation_files = [x for x in filelist if x.endswith('.phenotype.tif')]
validation_files.sort()

key_file = open('/scr/yren/wandb_key.txt', 'r')
key = key_file.readline()
wandb.login(key=key)

for i in range(len(validation_files)):
    imid = validation_files[i].split('.')[0]
    
    # Initialize WanDB
    wandb.init(
        project='Best_Experiment',
        config={
            "architecture":ARCHITECTURE,
            'model':'microSAM',
        },
        name=f'{imid}',
        reinit=True
    )
    
    im = skimage.io.imread(DIRECTORY + validation_files[i])
    im = ((im / im.max()) * 255).astype(np.uint8)
    
    # Document Inference Time
    s = time.time()
    model_choice = 'vit_b_lm'
    H,W = im.shape
    if (H > 1024) and (W > 1024):
        prediction = run_automatic_instance_segmentation(im, ndim=2, model_type=model_choice, device='cuda', tile_shape=(1024, 1024), halo=(256, 256))
    else:
        prediction = run_automatic_instance_segmentation(im, ndim=2, model_type=model_choice, device='cuda')
    e = time.time()
    wandb.log({'Inference Time': e-s})
    
    prediction = np.asarray(prediction, dtype='uint16')
    
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
    mn_gt = src.dinomn.mnds.read_image(DIRECTORY, imid, 'phenotype_outlines.png', scale=SCALE_FACTOR)
    src.dinomn.evaluation.segmentation_report(imid=imid, predictions=micro_mask, gt=mn_gt, intersection_ratio=0.1, report_obj='Micronuclei')

# release the resources
wandb.finish()