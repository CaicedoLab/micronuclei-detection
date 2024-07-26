from glob2 import glob
import sys
sys.path.append('../')
import skimage
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import csv
from microdet.evaluation import compare_two_labels, measures_at

def evaluate_cellpose(micronuclei_gt_path, micronuclei_mask_path):
    micronuclei_gt = skimage.io.imread(micronuclei_gt_path)
    micronuclei_gt = micronuclei_gt[:,:,0] > 0 # Use only the red channel
    micronuclei_gt = scipy.ndimage.binary_fill_holes(micronuclei_gt) ^ micronuclei_gt
    #Merge seg and outlines_img together, overlay the outlines_img on seg
    micronuclei_gt = skimage.measure.label(micronuclei_gt)
    # Not doing object edge detection as the objects start touching then
    micronuclei_gt = skimage.morphology.dilation(micronuclei_gt)  # recover object edge

    micronuclei_img = skimage.io.imread(micronuclei_mask_path, as_gray=True)
    image_stats = skimage.measure.regionprops(micronuclei_img)

    full_area = []
    area_label_number = []

    for i in range(len(image_stats)):
        if(image_stats[i].area > 300.0):
            area_label_number.append(image_stats[i].label)
        full_area.append(image_stats[i].area)

    for i in area_label_number:
        micronuclei_img[micronuclei_img == i] = 0
    unique_values = np.unique(micronuclei_img)
    # Create a dictionary to map the old values to the new values
    value_map = {value: i for i, value in enumerate(unique_values)}
    # Apply the mapping to the image
    micronuclei_img = np.vectorize(value_map.get)(micronuclei_img)

    nb_overdetection, nb_underdetection, mean_IoU, IoUs = compare_two_labels(micronuclei_gt, micronuclei_img, True, False)
    f1, prec, rec, TP, FP, FN = measures_at(0.1, IoUs)

    return str(micronuclei_gt_path.split('/')[-1].split('.')[0]), f1, prec, rec, TP, FP, FN


results = []

train_dir = "/scr/vidit/dataset_nuclei"
outlines_dir = "/scr/vidit/dataset_v2"
phenotype_names = glob(train_dir + '/*phenotype_outlines.png')

for phenotype_image in phenotype_names:
    phenotype_mask = phenotype_image.replace('phenotype_outlines.png', 'phenotype_cp_masks.png')
    result = evaluate_cellpose(phenotype_image, phenotype_mask)
    print(phenotype_image)
    results.append(result)


# Write results to CSV
csv_file = 'cellpose_nuclei_model_results.csv'
csv_columns = ['Image', 'F1', 'Precision', 'Recall', 'True Positive', 'False Positive', 'False Negative']

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_columns)
    writer.writerows(results)

print(f"Results saved to {csv_file}")