import warnings
warnings.filterwarnings("ignore")

import os
from glob import glob
from IPython.display import FileLink
from typing import Union, Tuple, Optional

import numpy as np
import imageio.v3 as imageio
from matplotlib import pyplot as plt
from skimage.measure import label as connected_components

import torch

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.util.util import get_random_colors

import micro_sam.training as sam_training
from micro_sam.sample_data import fetch_tracking_example_data, fetch_tracking_segmentation_data
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


batch_size = 1  # the training batch size
patch_shape = (256, 256)  # the size of patches for training
sampler = MinInstanceSampler(min_size=25)

raw_key, label_key = '*.phenotype.tif', '*.phenotype_outlines.png'
train_dir = '/scr/yren/micronuclei-detection/baselines/microSAM_finetune_data/train/'
val_dir = '/scr/yren/micronuclei-detection/baselines/microSAM_finetune_data/validation/'
train_instance_segmentation = True
train_segmentation_dir = train_dir
val_segmentation_dir = val_dir

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

n_objects_per_batch = 10  # the number of objects per batch that will be sampled
device = "cuda" if torch.cuda.is_available() else "cpu"  # the device/GPU used for training
n_epochs = 5  # how long we train (in epochs)

# The model_type determines which base model is used to initialize the weights that are finetuned.
# We use vit_b here because it can be trained faster. Note that vit_h usually yields higher quality results.
model_type = "vit_b_lm"

# The name of the checkpoint. The checkpoints will be stored in './checkpoints/<checkpoint_name>'
checkpoint_name = "sam_hela"

root_dir = '/scr/yren/micronuclei-detection/baselines/microSAM_finetune_data'

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