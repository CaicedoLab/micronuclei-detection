from glob2 import glob
import sys
sys.path.append('../')
import skimage
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import csv

model_save_path = "/scr/vidit/models"
image_save_path = "/scr/vidit/dataset_original"
save_dir = "/scr/vidit/dataset_original"

import subprocess
os.environ['MKL_THREADING_LAYER'] = 'GNU'

for model_name in os.listdir(model_save_path):
    file_name = model_name[6:] + ".phenotype.tif"
    file_path = image_save_path + "/" + file_name
    model_path = model_save_path + "/" + model_name
    command = [
    'cellpose',
    '--image_path', file_path,
    '--pretrained_model', model_path,
    '--chan', '0',
    '--chan2', '0',
    '--save_png',
    '--verbose',
    '--use_gpu']
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Output:", result.stdout)
        print("Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e.stderr}")
    except FileNotFoundError:
        print("Cellpose is not installed or not found in the PATH.")

