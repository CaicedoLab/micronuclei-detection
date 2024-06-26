import skimage
import glob
import os
from PIL import Image
import numpy as np
import glob
import cv2
import os

folder_img_path = "/scr/vidit/dataset_v2"
save_path = "/scr/vidit/dataset_nuclei_division"

def divide_image(folder_path, save_path, extension):
    # Ensure the save_path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for image_path in glob.glob(os.path.join(folder_path, f"*{extension}")):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read the image {image_path}")
            continue
        height, width, _ = img.shape
        sub_height = height // 10
        sub_width = width // 10
        base_name = os.path.basename(image_path).replace(extension, "")
        for i in range(10):
            for j in range(10):
                sub_img = img[i*sub_height:(i+1)*sub_height, j*sub_width:(j+1)*sub_width]
                save_filename = f"{base_name}_{i*10+j}{extension}"
                save_filepath = os.path.join(save_path, save_filename)
                cv2.imwrite(save_filepath, sub_img)
                print(f"Saved {save_filepath}")

# Run for different extensions
#divide_image(folder_img_path, save_path, ".png")
#divide_image(folder_img_path, save_path, ".tif")


'''
This function will look at all the images present in the following function and remove
all the images that are black. This is done because the images that are black are not
useful for training the model.
'''

folder_path = "/scr/vidit/dataset_nuclei_division"

def remove_empty_images(folder_path):
    counter = 0
    for image in glob.glob(folder_path + "/*.png"):
        img = Image.open(image)
        #print(image)
        if not img.convert('L').getbbox():
            os.remove(image)
            counter += 1
            image = image.replace("_outlines", "")
            image = image.replace(".png", ".tif")
            os.remove(image)
            image = image.replace("phenotype", "nuclei-clean")
            os.remove(image)
    print(f"Total number of images removed: {counter}")


def join_masks(png_file, tif_file):
    png_img = cv2.imread(png_file)
    tif_img = cv2.imread(tif_file)

    
