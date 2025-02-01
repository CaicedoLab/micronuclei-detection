import os
import sys
import torch
import mnmodel
import mnds
import evaluation
import numpy as np

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
EDGES = FALSE
ARCHITECTURE = "LOO with dataset v2 & v3"


CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/scale_aligned_v2_v3'
OUTPUT_DIR = "/model_output/output/"

# set CHTC writeable cahce directory for pytorch and matplotlib
os.environ['TORCH_HOME'] = CURRENT_PATH + '/.cache/torch'
os.environ['MPLCONFIGDIR'] = CURRENT_PATH + '/.cache/matplotlib/config'

if len(sys.argv) < 2:
    print("Use: python leave_one_out.py gpu")
    sys.exit()

    
# i = int(sys.argv[1])
gpu = int(sys.argv[1])
device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'


# Train
files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
annot_files = [x for x in filelist if x.endswith('.phenotype_outlines.png')]
annot_files.sort()


# for i in range(len(annot_files)):
for i in range(6):
# if True:
    # temp code
    hela_rpe_ids = ['C1-20X_c0_B3_Tile-15.phenotype_outlines.png', 
                'C1-20X_c0_B3_Tile-24.phenotype_outlines.png', 
                'C1-20X_c0_B3_Tile-5.phenotype_outlines.png', 
                'C1-20X_c0_phalloidin_C3_Tile-15.phenotype_outlines.png', 
                'C1-20X_c0_phalloidin_C3_Tile-18.phenotype_outlines.png', 
                'C1-20X_c0_phalloidin_C3_Tile-44.phenotype_outlines.png']
    
    training_files = annot_files.copy()
    # validation_files = [annot_files[i]]
    validation_files = [hela_rpe_ids[i]]
    j = annot_files.index(hela_rpe_ids[i])
    del training_files[j]
    # del training_files[i]

    image_id = validation_files[0].split('.')[0]

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
            'Number of training images':len(training_files)
        },
        name=f'{image_id}',
        reinit=True
    )

    # Create model
    model = mnmodel.MicronucleiModel(
        DIRECTORY, 
        device, 
        training_files=training_files, 
        validation_files=validation_files, 
        patch_size=PATCH_SIZE,
        scale_factor=SCALE_FACTOR,
        edges=EDGES, # FALSE
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
    model.save(outdir=OUTPUT_DIR, model_name=image_id)


    # Validate
    predictions_dir = DIRECTORY + OUTPUT_DIR
    models_dir = OUTPUT_DIR

    # Select image for analysis
    # validation_file = annot_files[i]
    validation_file = annot_files[j]
    imid = validation_file.split('.')[0]

    # Load image and annotations
    im = mnds.read_image(DIRECTORY, imid, 'phenotype.tif', scale=SCALE_FACTOR)
    im = np.array((im - np.min(im))/(np.max(im) - np.min(im)), dtype="float32")
    mn_gt = mnds.read_image(DIRECTORY, imid, 'phenotype_outlines.png', scale=SCALE_FACTOR)
    mn_gt = mn_gt > 0 # convert to boolean (binary mask)

    # Load model and compute probabilities
    model = mnmodel.MicronucleiModel(
        DIRECTORY, 
        device, 
        patch_size=PATCH_SIZE, 
        edges=EDGES,
        gaussian=False # predict() function has nothing to do with gaussian sampling
    )
    model.load(validation_file.replace('phenotype_outlines.png','pth'), model_dir=models_dir)
    probabilities = model.predict(im, stride=1, step=STEP, batch_size=BATCH_SIZE, dilation=DILATION)
    filename = predictions_dir + validation_file.replace('phenotype_outlines.png','_probabilities')
    np.save(filename, probabilities)

    mn_pred = probabilities[0,:,:] > THRESHOLD
    evaluation.segmentation_report(imid=imid, predictions=mn_pred, gt=mn_gt, intersection_ratio=0.1, report_obj='Micronuclei')

    # release the resources
    torch.cuda.empty_cache()
    wandb.finish()
