import os, shutil
os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = "./.cache"
os.environ["NUMBA_CACHE_DIR"] = "/tmp"
import numpy as np
from skimage.io import imread
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from cellpose import core, utils, io, models, metrics, train
from glob2 import glob

train_dir = "dataset_division"
save_path = "/staging/groups/caicedo_group/micronuclei/cellpose-models"

# Function to read a single pair of data and label images
def read_image_pair(file_path):
    try:
        if file_path.split("/")[-1].startswith("image_name"):
            return None, None
        data_img = imread(file_path)
        label_file_path = file_path.replace('phenotype.', 'phenotype_outlines.')
        label_img = imread(label_file_path)
        return data_img, label_img
    except Exception as e:
        print(f"Error reading files: {file_path}, {e}")
        return None, None

# Initialize lists to store data and labels
train_data = []
train_label = []

# Use ThreadPoolExecutor to parallelize file reading
file_paths = glob(os.path.join(train_dir, '*_phenotype.*'))

with ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(read_image_pair, file_paths), total=len(file_paths)))

for data_img, label_img in results:
    if data_img is not None and label_img is not None:
        train_data.append(data_img)
        train_label.append(label_img)

# Convert lists to numpy arrays
train_data = np.array(train_data)
train_label = np.array(train_label)

# Ensure the images are in N, C, H, W format
train_data = np.moveaxis(train_data, -1, 1)
train_label = np.expand_dims(train_label, axis=1)  # Add channel dimension (C=1)


# start logger (to see training across epochs)
#logger = io.logger_setup()
use_GPU = core.use_gpu()

initial_model = "nuclei" #@param ["cyto", "cyto3", "nuclei", "tissuenet_cp3", "livecell_cp3", "yeast_PhC_cp3", "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3", "deepbacs_cp3", "scratch"]
model_name = "model_image_name"

# DEFINE CELLPOSE MODEL (without size model)
model = models.CellposeModel(gpu=use_GPU, model_type=initial_model)

# set channels
channels = [0, 0]

Use_Default_Advanced_Parameters = True #@param {type:"boolean"}
learning_rate = 0.1 #@param {type:"number"}
weight_decay = 0.0001 #@param {type:"number"}

new_model_path = train.train_seg(model.net, train_data=train_data,
                              train_labels=train_label,
                              channels=channels,
                              save_path=save_path,
                              n_epochs=100,
                              learning_rate=0.1,
                              weight_decay=weight_decay,
                              SGD=True,
                              min_train_masks = 1,
                              normalize=True,
                              model_name=model_name)



# diameter of labels in training images
diam_labels = model.net.diam_labels.item()