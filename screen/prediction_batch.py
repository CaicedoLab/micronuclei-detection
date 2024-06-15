import os
import sys
import time
import torch
from skimage import io
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
import mnds
import mnmodel
import evaluation
os.environ['TORCH_HOME'] = './.cache'



DIRECTORY = ""
device = "cuda"

folder_path = "plate_number_images_idea"
file_names = os.listdir(folder_path)
#file_names = np.random.choice(file_names, 20, replace=False) 

SCALE_FACTOR = 1.0

PATCH_SIZE = 256
STRIDE = 8
FEATURE_SIZE = 384 # dinov2 vit small model
# FEATURE_SIZE = 768 # dinov2 vit base model
TOKENS_PER_PATCH = PATCH_SIZE // STRIDE
STEP = 16
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.0001

# Start timing the entire script
start_time = time.time()

def model_predict(model, im, stride=1, step=16, batch_size=32):
    probabilities = model.predict(im, stride=stride, step=step, batch_size=batch_size)
    return probabilities

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

save_folder = "output/"

# Loading image
for file_name in file_names:
    im = io.imread(os.path.join(folder_path, file_name))
    im = np.array((im - np.min(im))/(np.max(im) - np.min(im)), dtype="float32")
    probs = model_predict(model, im, stride=1, step=STEP, batch_size=BATCH_SIZE)
    np.save(os.path.join(save_folder, file_name.replace(".tif", ".npy")), probs)
