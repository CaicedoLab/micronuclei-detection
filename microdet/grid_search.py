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
WEIGHT_DECAY = 1e-6


CURRENT_PATH = os.getcwd()
DIRECTORY = CURRENT_PATH + '/dataset_v2'
OUTPUT_DIR = "/model_output/nuclei_experiments/"

# set CHTC writeable cahce directory for pytorch and matplotlib
os.environ['TORCH_HOME'] = CURRENT_PATH + '/.cache/torch'
os.environ['MPLCONFIGDIR'] = CURRENT_PATH + '/.cache/matplotlib/config'


if len(sys.argv) < 6:
    print("Use: python grid_search.py experiment_id loss_fn learning_rate batch_size finetune(True/False)")
    sys.exit()


# i = int(sys.argv[1])
experiment_id = int(sys.argv[1])
LOSS_FN = str(sys.argv[2])
LR = float(sys.argv[3])
BATCH_SIZE = int(sys.argv[4])
FINETUNE = eval(sys.argv[5]) # dont use bool, only eval turns string to boolean!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# gpu = sys.argv[7]
# device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'

os.makedirs('config_output')

# Train
files = os.listdir(DIRECTORY)
filelist = [file for file in files if not file.startswith('.')] # avoid files starting with . when untarring in CHTC
annot_files = [x for x in filelist if x.endswith('png')]
annot_files.sort()
annot_files = annot_files[0:10]

# print(" *** ", validation_files, " *** ")
# image_id = validation_files[0].split('.')[0].split('_')[-1]
wandb.login(key='b3f4f9254c123781af918799b27affa92d8f4eeb')
wandb.init(
    project='grid_search_finetune',
    config={
        "architecture":"3 blocks of one 2x2 upscale and one 3x3 conv layers",
        "Loss": LOSS_FN,
        "fine_tuning":FINETUNE,
        "batch_size":BATCH_SIZE,
        "learning_rate":LR,
        "epochs": EPOCHS,
        "feature_size":FEATURE_SIZE,
        "patch_size":PATCH_SIZE,
        "weight_decay":WEIGHT_DECAY,
        "probability_threshold":THRESHOLD
    },
    name=f'experiment{experiment_id}'
)

for i in range(len(annot_files)): # reduce the number of jobs on chtc to avoid crushing
    training_files = annot_files.copy()
    validation_files = [annot_files[i]]
    del training_files[i]

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
                weight_decay=WEIGHT_DECAY)

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
