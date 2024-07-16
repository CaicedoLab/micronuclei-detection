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


folder_img_path = "/scr/vidit/dataset_v2"
save_path = "/scr/vidit/dataset_sliding_division"

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
        if(base_name.split(".")[-1] == "nuclei-clean" or base_name.split(".")[-1] == "phenotype_outlines" or base_name.split(".")[-1] == "phenotype"):
            for i in range(10):
                for j in range(10):
                    sub_img = img[i*sub_height:(i+1)*sub_height, j*sub_width:(j+1)*sub_width]
                    save_filename = f"{base_name}_{i*10+j}{extension}"
                    save_filepath = os.path.join(save_path, save_filename)
                    cv2.imwrite(save_filepath, sub_img)


def divide_image_slide(folder_path, save_path, extension, window_size=296, stride=74):
    # Ensure the save_path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for image_path in glob.glob(os.path.join(folder_path, f"*{extension}")):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read the image {image_path}")
            continue
        height, width, _ = img.shape
        base_name = os.path.basename(image_path).replace(extension, "")
        if(base_name.split(".")[-1] == "nuclei-clean" or base_name.split(".")[-1] == "phenotype_outlines" or base_name.split(".")[-1] == "phenotype"):
            count = 0
            for i in range(0, height - window_size + 1, stride):
                for j in range(0, width - window_size + 1, stride):
                    sub_img = img[i:i+window_size, j:j+window_size]
                    left_side = base_name.split(".")[-2]
                    right_side = base_name.split(".")[-1]
                    save_filename = f"{left_side}_{count}_{right_side}{extension}"
                    save_filepath = os.path.join(save_path, save_filename)
                    print(save_filepath)
                    cv2.imwrite(save_filepath, sub_img)
                    count += 1

#divide_image(folder_img_path, save_path, ".png", window_size=296, stride=74)
#divide_image(folder_img_path, save_path, ".tif", window_size=296, stride=74)

'''
This function will look at all the images present in the following function and remove
all the images that are black. This is done because the images that are black are not
useful for training the model.
'''

folder_path = "/scr/vidit/dataset_sliding_division"

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

#remove_empty_images(folder_path)


def remove_empty_nuclei_masks(folder_path):
    counter = 0
    for image in glob.glob(folder_path + "/*nuclei-clean*"):
        img = Image.open(image)
        if not img.convert('L').getbbox():
            os.remove(image)
            counter += 1
            image = image.replace("nuclei-clean", "phenotype")
            os.remove(image)
            image = image.replace("phenotype", "phenotype_outlines")
            image = image.replace(".tif", ".png")
            os.remove(image)
    print(f"Total number of images removed: {counter}")

#remove_empty_nuclei_masks(folder_path)

'''
This function will take the nuclei masks and the outlines masks and merge them together 
to have one label for them making it easier to feed it into cellpose.
'''
def join_masks(tif_path, outlines_path):
    tif_img = skimage.io.imread(tif_path, as_gray=True)
    outlines_img = skimage.io.imread(outlines_path)

    outlines_img = outlines_img[:,:,0] > 0 # Use only the red channel
    outlines_img = scipy.ndimage.binary_fill_holes(outlines_img) ^ outlines_img
    tif_img[outlines_img] = 1
    seg = tif_img > 0
    #Merge seg and outlines_img together, overlay the outlines_img on seg
    labels = skimage.measure.label(seg)
    # Not doing object edge detection as the objects start touching then
    #labels = skimage.morphology.dilation(labels)  # recover object edge
    return labels

'''

'''
def get_masks(folder_path):
    for image in glob.glob(folder_path + "/*phenotype_outlines*"):
        outlines_path = image
        image = image.replace("phenotype_outlines", "nuclei-clean")
        image = image.replace(".png", ".tif")
        tif_image = image
        labels = join_masks(tif_image, outlines_path)
        image = image.replace("nuclei-clean", "combined_outlines")
        skimage.io.imsave(image, labels)


#get_masks(folder_path)

def change_label_names(folder_path):
    for image_path in glob.glob(folder_path + "*combined_outlines*"):
        orig_path = image_path
        num = image_path.split(".")[-2].split("_")[-1]
        mid_string = image_path.split(".")[-2]
        mid_string = mid_string.replace(f"combined_outlines_{num}", "phenotype_outlines")
        image_path = image_path.split(".")[0] + f".{num}_" + mid_string + ".tif"
        #print(image_path)
        os.rename(orig_path, image_path)
    for image_path in glob.glob(folder_path + "*phenotype*"):
        orig_path = image_path
        num = image_path.split(".")[-2].split("_")[-1]
        mid_string = image_path.split(".")[-2]
        mid_string = mid_string.replace(f"_{num}", "")
        image_path = image_path.split(".")[0] + f".{num}_" + mid_string + ".tif"
        #print(image_path)
        #os.rename(orig_path, image_path)


#change_label_names("/scr/vidit/dataset_nuclei_division/test/")
