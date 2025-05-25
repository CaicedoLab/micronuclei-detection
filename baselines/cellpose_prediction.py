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

CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/all_data_micronuclei_no_rescale/validation/'

SCALE_FACTOR = 1.0

if len(sys.argv) < 2:
    print("Use: python cellpose_prediction.py specialist")
    sys.exit()

specialist = str(sys.argv[1])

ARCHITECTURE = f"Cellpose specialist - {specialist}"

# Train
files = os.listdir(DIRECTORY.replace('validation', 'train'))
filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
annot_files = [x for x in filelist if x.endswith('.phenotype_outlines.png')]
annot_files.sort()
training_files = annot_files.copy()

df = pd.read_csv(CURRENT_PATH + '/all_data_micronuclei_no_rescale/metadata.csv')
if specialist == 'frozen_all':
    files_to_keep = df[df.split == 'train'].filenames.to_list() # keep all training files
elif specialist == 'finetune_all':
    files_to_keep = df[df.split == 'train'].filenames.to_list() # keep all training files
elif specialist == 'pilot_screen':
    files_to_keep = df[(df.datasets.isin(['pilot', 'screen'])) & (df.split == 'train')].filenames.to_list()
elif specialist == 'hela_rpe1':
    files_to_keep = df[(df.datasets.isin(['HeLa', 'RPE1'])) & (df.split == 'train')].filenames.to_list()
else:
    files_to_keep = df[(df.datasets.isin([specialist])) & (df.split == 'train')].filenames.to_list()

fn = lambda file: file.replace('phenotype.tif', 'phenotype_outlines.png')
new_training_files = [fn(file) for file in files_to_keep]
fn2 = lambda file: file.replace('phenotype_outlines.png', 'phenotype.tif')
inputs = [fn2(file) for file in new_training_files]
    

files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
validation_files = [x for x in filelist if x.endswith('.phenotype.tif')]
validation_files.sort()

key_file = open('/scr/yren/wandb_key.txt', 'r')
key = key_file.readline()
wandb.login(key=key)

path = CURRENT_PATH + '/all_data_micronuclei_no_rescale/train/'
imgs = [io.imread(path + f) for f in inputs]
gts = [io.imread(path + f) for f in new_training_files]

if specialist == 'frozen_all':
    model = models.CellposeModel(gpu=True, model_type='cyto3')
    channels = [0,0]
else:
    model = models.CellposeModel(gpu=True, model_type='cyto3')
    channels = [0,0]

    save_path = '/scr/yren/micronuclei-detection/baselines/cellpose_models/'
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=imgs,
        train_labels=gts,
        channels=[0, 0],              # Use first channel (grayscale)
        channel_axis=None,            # No channel dimension in your data
        weight_decay=1e-6,
        learning_rate=1e-5,
        n_epochs=100,
        normalize=True,
        min_train_masks=0,
        model_name=f"cellpose_{specialist}_specialist.pth",
        save_path=save_path
    )


MICRON_AREA_THRESHOLD = 300 # 300 is the best cut-off
for i in range(len(validation_files)):
    imid = validation_files[i].split('.')[0]
    
    if specialist == 'frozen_all':
        wandb.init(
            project='Best_Experiment',
            config={
                "architecture":ARCHITECTURE,
                'model':'Cellpose cyto3',
                'area_threshold':MICRON_AREA_THRESHOLD,
                # 'diameter':diam_labels,
                'num of training images': len(new_training_files)
            },
            name=f'{imid}',
            reinit=True
        )
    else:
        diam_labels = model.diam_labels.copy()
        wandb.init(
            project='Best_Experiment',
            config={
                "architecture":ARCHITECTURE,
                'model':'Cellpose cyto3',
                'area_threshold':MICRON_AREA_THRESHOLD,
                'diameter':diam_labels,
                'num of training images': len(new_training_files)
            },
            name=f'{imid}',
            reinit=True
        )
    
    im = skimage.io.imread(DIRECTORY + validation_files[i])
    
    # Document inference time
    s = time.time()
    if specialist == 'frozen_all':
        masks = model.eval(im, channels=channels)[0]
    else:
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