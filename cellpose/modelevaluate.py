import skimage
import glob
import os
from PIL import Image
import numpy as np
import glob
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage
import scipy
import preprocess

img_path = "/home/MORGRIDGE/vagrawal/vidit/cellposeautomate/dataset"
Dataset_size = ["Small", "Big"]
Model_type = ["Scratch", "Finetuned", "Nuclei"]

'''
This function will make the appropriate train and test directories so that we can use
cellpose training functions after that
'''

def leave_one_out(img_path, filename):
    filenames = glob.glob(os.path.join(img_path, "*phenotype.*"))
    if not os.path.exists(os.join(img_path, "train")):
        os.mkdir(os.join(img_path, "train"))
    if not os.path.exists(os.join(img_path, "test")):
        os.mkdir(os.join(img_path, "test"))


def preprocess_image(dataset_size, model_type, img_path):
    leave_one_out(img_path)
    # go one directory behind in img_path
    os.chdir(os.pardir)
    if not os.path.exists(f"{dataset_size}_{model_type}"):
        os.mkdir(f"{dataset_size}_{model_type}")
    os.chdir(f"{dataset_size}_{model_type}")
    print(os.getcwd())
    '''
    if dataset_size == "Small":
        preprocess.divide_image(img_path, os.getcwd(), ".png")
        preprocess.divide_image(img_path, os.getcwd(), ".tif")

    elif dataset_size == "Big":
        preprocess.divide_image_slide(img_path, os.getcwd(), ".png")
        preprocess.divide_image_slide(img_path, os.getcwd(), ".tif")
    '''

preprocess_image("Small", "Scratch", img_path)




