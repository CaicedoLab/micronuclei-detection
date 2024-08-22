'''
Train Best Model with Leave-One-Out Strategy
'''
import os
import sys
import torch
import mnmodel
import os
import sys
import torch
import numpy as np
import mnds
import mnmodel
import evaluation
import wandb

SCALE_FACTOR = 1.0
PATCH_SIZE = 256
STRIDE = 8
FEATURE_SIZE = 384
TOKENS_PER_PATCH = PATCH_SIZE // STRIDE
STEP = 16
EPOCHS = 20
THRESHOLD = 0.5

LOSS_FN = 'combined'
LR = 1e-5
BATCH_SIZE = 32
FINETUNE = True
WEIGHT_DECAY = 1e-6


CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/dataset_v2'
OUTPUT_DIR = "/model_output/output/"

# set CHTC writeable cahce directory for pytorch and matplotlib
os.environ['TORCH_HOME'] = CURRENT_PATH + '/.cache/torch'
os.environ['MPLCONFIGDIR'] = CURRENT_PATH + '/.cache/matplotlib/config'

if len(sys.argv) < 2:
    print("Use: python run_best_experiment.py imidx")
    sys.exit()
    
i = int(sys.argv[1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs('best_experiment')

# Train
files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
annot_files = [x for x in filelist if x.endswith('png')]
annot_files.sort()
# annot_files = annot_files[0:10] # using all 18 images


training_files = annot_files.copy()
validation_files = [annot_files[i]]
del training_files[i]

lst = validation_files[0].split('.')[0].split('_')[-2:]
image_id = f'{lst[0]}-{lst[1]}'
wandb.login(key='')
wandb.init(
    project='Best_Experiment',
    config={
        "architecture":"2 upscale followed with 3 blocks of conv, norm, relu and skip connections",
        "Loss": LOSS_FN,
        "Loss Weight": "all default, sam ratio (0.95focal+0.05dice) + gamma=2, etc",
        "fine_tuning":FINETUNE,
        "batch_size":BATCH_SIZE,
        "learning_rate":LR,
        "epochs": EPOCHS,
        "feature_size":FEATURE_SIZE,
        "patch_size":PATCH_SIZE,
        "weight_decay":WEIGHT_DECAY,
        "probability_threshold":THRESHOLD
    },
    name=f'1st-{image_id}'
)

# Create model
model = mnmodel.MicronucleiModel(
    DIRECTORY, 
    device, 
    training_files=training_files, 
    validation_files=validation_files, 
    patch_size=PATCH_SIZE,
    scale_factor=SCALE_FACTOR,
    edges=True
)

# Train
model.train(epochs=EPOCHS, 
            batch_size=BATCH_SIZE, 
            learning_rate=LR, 
            loss_fn=LOSS_FN, 
            output_dir=OUTPUT_DIR, 
            finetune=FINETUNE,
            weight_decay=WEIGHT_DECAY
            )

# Save
model.save(outdir=OUTPUT_DIR)


# Validate
predictions_dir = DIRECTORY + OUTPUT_DIR
models_dir = OUTPUT_DIR

# Select image for analysis
validation_file = annot_files[i]
imid = validation_file.split('.')[0]

# Load image and annotations
im = mnds.read_image(DIRECTORY, imid, 'phenotype.tif', scale=SCALE_FACTOR)
im = np.array((im - np.min(im))/(np.max(im) - np.min(im)), dtype="float32")
mn_gt = mnds.read_micronuclei_masks(DIRECTORY, imid, SCALE_FACTOR)

# Load model and compute probabilities
model = mnmodel.MicronucleiModel(DIRECTORY, device, patch_size=PATCH_SIZE, edges=True)
model.load(validation_file.replace('phenotype_outlines.png','pth'), model_dir=models_dir)
probabilities = model.predict(im, stride=1, step=STEP, batch_size=BATCH_SIZE)
filename = predictions_dir + validation_file.replace('phenotype_outlines.png','_probabilities')
np.save(filename, probabilities)

mn_pred = probabilities[0,:,:] > THRESHOLD
evaluation.segmentation_report(imid=imid, predictions=mn_pred, gt=mn_gt, intersection_ratio=0.1, report_obj='Micronuclei')

# release the resources
torch.cuda.empty_cache()
wandb.finish()
