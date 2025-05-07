import os
import sys
import torch
import mnmodel
import os
import sys
import torch
import numpy as np
from dinomn import mnds
from dinomn import mnmodel
from dinomn import evaluation
import wandb
from tqdm import tqdm
import skimage

# Fixed Hyperparameters
PATCH_SIZE = 256
FEATURE_SIZE = 384
EPOCHS = 20
THRESHOLD = 0.5
IoU_THRESHOLD = 0.1 # for micronuclei

LOSS_FN = 'combined'
FINETUNE = True
WEIGHT_DECAY = 1e-6

# Tunable Hyperparameters
SCALE_FACTOR = 1.0
GAUSSIAN = True
EDGES = False
ARCHITECTURE = "DinoMN: Grid Search with new architecture"

STEP = 16
DILATION = 2 # the best, only used in prediction
OVERSAMPLE = 'NO'


CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/all_data_micronuclei_no_rescale/train'
OUTPUT_DIR = "/model_output/grid_search/"

# set CHTC writeable cahce directory for pytorch and matplotlib
os.environ['TORCH_HOME'] = CURRENT_PATH + '/.cache/torch'
os.environ['MPLCONFIGDIR'] = CURRENT_PATH + '/.cache/matplotlib/config'
torch.set_num_threads(8)

# if len(sys.argv) < 6:
#     print("Use: python grid_search.py experiment_id loss_fn learning_rate batch_size finetune(True/False)")
#     sys.exit()

if len(sys.argv) < 4:
    print("Use: python grid_search.py experiment_id learning_rate batch_size")
    sys.exit()


# i = int(sys.argv[1])
experiment_id = int(sys.argv[1])
# LOSS_FN = str(sys.argv[2])
LR = float(sys.argv[2])
BATCH_SIZE = int(sys.argv[3])
# FINETUNE = eval(sys.argv[5]) # dont use bool, only eval turns string to boolean!
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Train
files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
annot_files = [x for x in filelist if x.endswith('png')]
annot_files.sort()

training_files = annot_files.copy()

# Validation Files
validation_files = os.listdir(CURRENT_PATH + '/all_data_micronuclei_no_rescale/validation')
validation_files = [file for file in validation_files if not file.startswith('.')]
validation_files = [x for x in validation_files if x.endswith('.phenotype_outlines.png')]
validation_files.sort()

validation_filelist = validation_files.copy()


key_file = open('./wandb_key.txt', 'r')
key = key_file.readline()
wandb.login(key=key)
wandb.init(
    project='Grid-Search-Micronuclei',
    config={
        "architecture":ARCHITECTURE,
        "Loss": LOSS_FN,
        "Loss Weight": "all default, sam ratio (0.95focal+0.05dice) + gamma=2, etc",
        "fine_tuning":FINETUNE,
        "batch_size":BATCH_SIZE,
        "start_learning_rate":LR,
        "lr_scheduler":"Cosine",
        "scale_factor":'Trained on non-scaled images',
        "epochs": EPOCHS,
        "feature_size":FEATURE_SIZE,
        "patch_size":PATCH_SIZE,
        "weight_decay":WEIGHT_DECAY,
        "probability_threshold":THRESHOLD,
        "gaussian":GAUSSIAN,
        'edges':EDGES,
        'Number of training images':len(training_files),
        'Number of validation images':len(validation_filelist),
        'Oversample':OVERSAMPLE
    },
    # concate lr and batch size here
    name=f'experiment{experiment_id}: {LR}-{BATCH_SIZE}'
)

# Create model
model = mnmodel.MicronucleiModel(
    DIRECTORY, 
    device, 
    training_files=training_files, 
    validation_files=validation_filelist,
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
            weight_decay=WEIGHT_DECAY
)

# Save
model.save(outdir=OUTPUT_DIR, model_name='DinoMN')


################ Making Predictions ################
# Validate
predictions_dir = DIRECTORY + OUTPUT_DIR
models_dir = OUTPUT_DIR

# Load model and compute probabilities
model = mnmodel.MicronucleiModel(
    data_dir=CURRENT_PATH + '/all_data_micronuclei_no_rescale/train',
    device=device
)
# model.load(validation_file.replace('phenotype_outlines.png','pth'), model_dir=models_dir)
model.load('DinoMN.pth', model_dir=models_dir)

for i in tqdm(range(len(validation_filelist))):
# Select image for analysis
    validation_file = validation_filelist[i]
    imid = validation_file.split('.')[0]
    
    wandb.init(
        project='Grid-Search-Micronuclei',
        config={
            "architecture":f'experiment{experiment_id}: {LR}-{BATCH_SIZE}',
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
            "IoU_threshold":IoU_THRESHOLD,
            "dilation":DILATION,
            "gaussian":'gaussian not need for prediction',
            'Number of validation images':len(annot_files)
        },
        name=f'{imid}',
        reinit=True
    )
    

    # Load image and annotations
    DIRECTORY = CURRENT_PATH + '/all_data_micronuclei_no_rescale/validation'
    im = mnds.read_image(DIRECTORY, imid, 'phenotype.tif', scale=SCALE_FACTOR)
    
    im = np.array((im - np.min(im))/(np.max(im) - np.min(im)), dtype="float32")
    mn_gt = mnds.read_image(DIRECTORY, imid, 'phenotype_outlines.png', scale=SCALE_FACTOR)
    mn_gt = mn_gt > 0

    probabilities = model.predict(im, stride=1, step=STEP, batch_size=BATCH_SIZE)
    filename = predictions_dir + validation_file.replace('phenotype_outlines.png','_probabilities')
    

    mn_pred = probabilities[0,:,:] > THRESHOLD
    labeled_mn = skimage.morphology.label(mn_pred)
    labeled_mn = np.asarray(labeled_mn, dtype='uint16') # if saving as img
    
    # dilate the labeled mn
    if DILATION > 0:
        dilation = skimage.morphology.disk(DILATION)
        labeled_mn = skimage.morphology.dilation(labeled_mn, dilation)
        

    evaluation.segmentation_report(imid=imid, predictions=mn_pred, gt=mn_gt, intersection_ratio=IoU_THRESHOLD, report_obj='Micronuclei')

    # save labeled matrices
    # np.save(filename, labeled_mn)
    
# release the resources
torch.cuda.empty_cache()
wandb.finish()
