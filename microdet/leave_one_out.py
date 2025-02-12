import os
import sys
import torch
import mnmodel
import mnds
import evaluation
import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import sys
import wandb

# Fixed Hyperparameters
PATCH_SIZE = 256
FEATURE_SIZE = 384
STEP = 16
EPOCHS = 20
THRESHOLD = 0.5

LOSS_FN = 'combined'
LR = 1e-5
BATCH_SIZE = 32
FINETUNE = True
WEIGHT_DECAY = 1e-6

# Tunable Hyperparameters
SCALE_FACTOR = 1.0 # All images have been pre-scaled
DILATION = 2 # 2 might be the best, only affect inference
GAUSSIAN = True # only affect training
EDGES = True # if do LOO on v2 and v3 datasets only
ARCHITECTURE = "Non-scaled leave one microscope out experiment"


CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/all_data_micronuclei_no_rescale/train'
OUTPUT_DIR = "/model_output/output/"

# set CHTC writeable cahce directory for pytorch and matplotlib
os.environ['TORCH_HOME'] = CURRENT_PATH + '/.cache/torch'
os.environ['MPLCONFIGDIR'] = CURRENT_PATH + '/.cache/matplotlib/config'

if len(sys.argv) < 3:
    print("Use: python leave_one_out.py microscope gpu")
    sys.exit()

    
# i = int(sys.argv[1])
subset = str(sys.argv[1]) # microscope
gpu = int(sys.argv[2])
device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'


# Train
files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
annot_files = [x for x in filelist if x.endswith('.phenotype_outlines.png')]
annot_files.sort()
training_files = annot_files.copy()

# remove subset from training_files (NEED TO FIX CODE: WRONG)
df = pd.read_csv(CURRENT_PATH + '/all_data_micronuclei_no_rescale/metadata.csv')
files_to_remove = df[df.datasets == subset].filenames.to_list()
fn = lambda file: file.replace('phenotype.tif', 'phenotype_outlines.png')
files_to_remove = [fn(file) for file in files_to_remove]
new_training_files = [file for file in training_files if file not in files_to_remove]

# Validation Files
validation_files = os.listdir(CURRENT_PATH + '/all_data_micronuclei_no_rescale/validation')
validation_files = [file for file in validation_files if not file.startswith('.')]
validation_files = [x for x in validation_files if x.endswith('.phenotype_outlines.png')]
validation_files.sort()

validation_filelist = validation_files.copy()

# for i in range(len(annot_files)):
# for i in range(6):
if True:
    
    # training_files = annot_files.copy()
    # validation_files = [annot_files[i]]
    # del training_files[i]


    key_file = open('./wandb_key.txt', 'r')
    key = key_file.readline()
    wandb.login(key=key)
    wandb.init(
        project='Best_Experiment',
        config={
            "architecture":ARCHITECTURE,
            "Loss": LOSS_FN,
            "Loss Weight": "all default, sam ratio (0.95focal+0.05dice) + gamma=2, etc",
            "fine_tuning":FINETUNE,
            "batch_size":BATCH_SIZE,
            "start_learning_rate":LR,
            "lr_scheduler":"Cosine",
            "scale_factor":SCALE_FACTOR,
            "epochs": EPOCHS,
            "feature_size":FEATURE_SIZE,
            "patch_size":PATCH_SIZE,
            "weight_decay":WEIGHT_DECAY,
            "probability_threshold":THRESHOLD,
            "dilation":DILATION,
            "gaussian":GAUSSIAN,
            'edges':EDGES,
            'step':STEP,
            'Number of training images':len(new_training_files)
        },
        name=f'(Train) Leave {subset} out',
        reinit=True
    )

    # Create model
    model = mnmodel.MicronucleiModel(
        DIRECTORY, 
        device, 
        training_files=new_training_files, 
        validation_files=validation_files, 
        patch_size=PATCH_SIZE,
        scale_factor=SCALE_FACTOR,
        edges=EDGES, # False
        gaussian=GAUSSIAN # Gaussian is only applied in training stage
    )

    # Train
    model.train(epochs=EPOCHS, 
                batch_size=BATCH_SIZE, 
                learning_rate=LR, 
                loss_fn=LOSS_FN, 
                finetune=FINETUNE,
                weight_decay=WEIGHT_DECAY
                )

    # Save
    model.save(outdir=OUTPUT_DIR, model_name=f'leave_{subset}_out')
    
# Load model and compute probabilities
model = mnmodel.MicronucleiModel(
    DIRECTORY, 
    device
)
# model.load(validation_file.replace('phenotype_outlines.png','pth'), model_dir=models_dir)
model.load(f'leave_{subset}_out.pth', model_dir=OUTPUT_DIR)

# Validate
DIRECTORY = CURRENT_PATH + '/all_data_micronuclei_no_rescale/validation'
predictions_dir = DIRECTORY + OUTPUT_DIR
models_dir = OUTPUT_DIR

for i in tqdm(range(len(validation_files))):

    # Select image for analysis
    # validation_file = annot_files[i]
    validation_file = validation_files[i]
    imid = validation_file.split('.')[0]
    
    ARCHITECTURE = f'(Nonscaled) Eval 47 images leaving {subset} out'
    wandb.init(
        project='Best_Experiment',
        config={
            "architecture":ARCHITECTURE,
            "Loss": LOSS_FN,
            "Loss Weight": "all default, sam ratio (0.95focal+0.05dice) + gamma=2, etc",
            "fine_tuning":FINETUNE,
            "batch_size":BATCH_SIZE,
            "learning_rate":LR,
            "scale_factor":'Predict on images that are not scaled',
            "epochs": EPOCHS,
            'step':STEP,
            "feature_size":FEATURE_SIZE,
            "patch_size":PATCH_SIZE,
            "weight_decay":WEIGHT_DECAY,
            "probability_threshold":THRESHOLD,
            "dilation":DILATION,
            "gaussian":'gaussian not need for prediction',
            'Number of validation images':len(annot_files)
        },
        name=f'{imid}',
        reinit=True
    )

    # Load image and annotations
    im = mnds.read_image(DIRECTORY, imid, 'phenotype.tif', scale=SCALE_FACTOR)
    im = np.array((im - np.min(im))/(np.max(im) - np.min(im)), dtype="float32")
    mn_gt = mnds.read_image(DIRECTORY, imid, 'phenotype_outlines.png', scale=SCALE_FACTOR)
    mn_gt = mn_gt > 0 # convert to boolean (binary mask)

    probabilities = model.predict(im, stride=1, step=STEP, batch_size=BATCH_SIZE, dilation=DILATION)
    filename = predictions_dir + validation_file.replace('phenotype_outlines.png','_probabilities')
    np.save(filename, probabilities)

    mn_pred = probabilities[0,:,:] > THRESHOLD
    evaluation.segmentation_report(imid=imid, predictions=mn_pred, gt=mn_gt, intersection_ratio=0.1, report_obj='Micronuclei')


# release the resources
torch.cuda.empty_cache()
wandb.finish()
