import os
import sys
import time
import torch
from skimage import io
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'microdet')))

import mnds
import mnmodel
import evaluation

DIRECTORY = "/scr/yren/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCALE_FACTOR = 1.0

PATCH_SIZE = 256
STRIDE = 8
FEATURE_SIZE = 384 # dinov2 vit small model
# FEATURE_SIZE = 768 # dinov2 vit base model
TOKENS_PER_PATCH = PATCH_SIZE // STRIDE
STEP = 16
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.0001

# Start timing the entire script
start_time = time.time()

model = mnmodel.MicronucleiModel(
    data_dir = DIRECTORY,
    device = device,
    edges=True
)

# Timing the model loading
model_load_start = time.time()
model.load("best_model.pth", model_dir="")
model_load_end = time.time()

print(f"Model loading time: {model_load_end - model_load_start} seconds")

# Load model and compute probabilities
model = mnmodel.MicronucleiModel(DIRECTORY, device, patch_size=PATCH_SIZE, edges=True)
model.load("best_model.pth", model_dir="")

save_folder = ""

# Loading image
file = "/scr/vidit/screen-data/plate-1-channel-one/20X_c0-DAPI-GFP_A2_Tile-1.phenotype.tif"

# Timing the image loading
image_load_start = time.time()
im = io.imread(file)
image_load_end = time.time()

print(f"Image loading time: {image_load_end - image_load_start} seconds")

# Timing the prediction
prediction_start = time.time()
probabilities = model.predict(im, stride=1, step=STEP, batch_size=BATCH_SIZE)
prediction_end = time.time()

print(f"Prediction time: {prediction_end - prediction_start} seconds")

filename = "/scr/vidit/screen-data-probability-map/plate-1-channel-one/20X_c0-DAPI-GFP_A2_Tile-1.phenotype.npy"

# Timing the saving process
save_start = time.time()
np.save(filename, probabilities)
save_end = time.time()

print(f"Saving time: {save_end - save_start} seconds")

# End timing the entire script
end_time = time.time()

print(f"Total script time: {end_time - start_time} seconds")
