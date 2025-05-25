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

import dinomn.mnds as mnds
import dinomn.evaluation as evaluation

import wandb
import time


CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/all_data_micronuclei_no_rescale/validation/'
SCALE_FACTOR = 1.0

if len(sys.argv) < 2:
    print("Use: python microsam_prediction.py specialist")
    sys.exit()

specialist = str(sys.argv[1])


ARCHITECTURE = f"microSAM specialist - {specialist}"
# ARCHITECTURE = f"microSAM generalist finetuned"

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

# Data Loader
raw_key, label_key = '*.phenotype.tif', '*.phenotype_outlines.png'

# Generalist
if specialist == 'all':
    train_dir = './microSAM_generalist_data/train/'
    val_dir = './microSAM_generalist_data/validation/'
else:
    # Specialist
    train_dir = f'./microSAM_specialist_data/train/{specialist}/'
    if specialist == 'mnfinder_train':
        val_dir = f'./microSAM_specialist_data/validation/{specialist.replace('train', 'validation')}/'
    else:
        val_dir = f'./microSAM_specialist_data/validation/{specialist}/'
# print(f'Is validation directory valid: {os.path.isdir(val_dir)}')

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
device = "cuda" if torch.cuda.is_available() else "cpu"  # the device/GPU used for training
n_epochs = 5
model_type = "vit_b_lm"
checkpoint_name = f"sam_finetune_{specialist}"

if specialist == 'all':
    best_checkpoint = os.path.join(os.getcwd(), 'microSAM_generalist_data', "models", "checkpoints", checkpoint_name, "best.pt")
    root_dir = './microSAM_generalist_data'
else:
    best_checkpoint = os.path.join(os.getcwd(), 'microSAM_specialist_data', "models", "checkpoints", checkpoint_name, "best.pt")
    root_dir = './microSAM_specialist_data'

sam_training.train_sam(
    name=checkpoint_name,
    save_root=os.path.join(root_dir, "models"),
    model_type=model_type,
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=n_epochs,
    n_objects_per_batch=n_objects_per_batch,
    with_segmentation_decoder=train_instance_segmentation,
    device=device
)

# Validation
files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
validation_files = [x for x in filelist if x.endswith('.phenotype.tif')]
validation_files.sort()

key_file = open('/scr/yren/wandb_key.txt', 'r')
key = key_file.readline()
wandb.login(key=key)


# Load the best checkpoint
best_checkpoint = os.path.join(os.getcwd(), 'microSAM_specialist_data', "models", "checkpoints", checkpoint_name, "best.pt")
MICRON_AREA_THRESHOLD = 300
for i in range(len(validation_files)):
    imid = validation_files[i].split('.')[0]
    
    # Initialize WanDB
    wandb.init(
        project='Best_Experiment',
        config={
            "architecture":ARCHITECTURE,
            'model':f'microSAM finetune {specialist}',
            'area_threshold':MICRON_AREA_THRESHOLD
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
    
    save_path = CURRENT_PATH + '/microSAM_predictions/finetune/'
    np.save(save_path + imid + '._probabilities.npy', micro_mask)
    
    # evaluation
    mn_gt = mnds.read_image(DIRECTORY, imid, 'phenotype_outlines.png', scale=SCALE_FACTOR)
    evaluation.segmentation_report(imid=imid, predictions=micro_mask, gt=mn_gt, intersection_ratio=0.1, report_obj='Micronuclei')

# release the resources
wandb.finish()