import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import sys
sys.path.append('../')
import skimage
import wandb

import dinomn.evaluation as evaluation
import dinomn.mnds as mnds
from cellpose import io, models, train

ARCHITECTURE = "3rd Cellpose predictions - finetune"
CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/all_data_micronuclei_no_rescale/validation/'

SCALE_FACTOR = 1.0

# path = '/scr/yren/micronuclei-detection/baselines/all_data_micronuclei_no_rescale/'
# train_dir = os.path.join(path, 'train')
# val_dir = os.path.join(path, 'validation')

# output = io.load_train_test_data(train_dir, val_dir, image_filter=".phenotype",
#                                 mask_filter=".phenotype_outlines", look_one_level_down=False)
# images, labels, image_names, test_images, test_labels, image_names_test = output


files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
validation_files = [x for x in filelist if x.endswith('.phenotype.tif')]
validation_files.sort()

key_file = open('/scr/yren/wandb_key.txt', 'r')
key = key_file.readline()
wandb.login(key=key)

model = models.CellposeModel(gpu=True, model_type='cyto3')
channels = [0,0]

# save_path = '/scr/yren/micronuclei-detection/baselines/cellpose_models/'
# model_path, train_losses, test_losses = train.train_seg(
#     model.net,
#     train_data=images,
#     train_labels=labels,
#     channels=[0, 0],              # Use first channel (grayscale)
#     channel_axis=None,            # No channel dimension in your data
#     weight_decay=1e-6,
#     learning_rate=1e-5,
#     n_epochs=100,
#     normalize=True,
#     model_name="finetuned_cellpose.pth",
#     save_path=save_path
# )

MICRON_AREA_THRESHOLD = 100

for i in range(len(validation_files)):
    imid = validation_files[i].split('.')[0]
    
    diam_labels = model.diam_labels.copy()
    
    wandb.init(
        project='Best_Experiment',
        config={
            "architecture":ARCHITECTURE,
            'model':'Cellpose cyto3',
            'area_threshold':MICRON_AREA_THRESHOLD,
            'diameter':diam_labels
        },
        name=f'{imid}',
        reinit=True
    )
    
    im = skimage.io.imread(DIRECTORY + validation_files[i])
    
    # Document inference time
    s = time.time()
    masks = model.eval(im, channels=channels, diameter=diam_labels)[0]
    e = time.time()
    wandb.log({'Inference Time': e-s})
    
    masks = np.asarray(masks, dtype='uint16')
    
    labels = skimage.morphology.label(masks)
    micron_labels = []
    for i in range(1, len(np.unique(labels))):
        area = np.sum(labels == i)
        if area < MICRON_AREA_THRESHOLD:
            micron_labels.append(i)
            
    micro_mask = np.zeros_like(masks)
    for i in micron_labels:
        micro_mask += (labels == i)
        
    save_path = CURRENT_PATH + '/cellpose_predictions/'
    np.save(save_path + imid + '._probabilities.npy', micro_mask)
    
    # evaluation
    mn_gt = mnds.read_image(DIRECTORY, imid, 'phenotype_outlines.png', scale=SCALE_FACTOR)
    evaluation.segmentation_report(imid=imid, predictions=micro_mask, gt=mn_gt, intersection_ratio=0.1, report_obj='Micronuclei')

# release the resources
wandb.finish()