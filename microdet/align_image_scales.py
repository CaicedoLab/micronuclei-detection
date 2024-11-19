#!/usr/bin/env python
# coding: utf-8

# Align images that are produced by different microscopes
# Finally: align dataset_v2, v_3, mnfinder train test, BBBC038 data and put them togerther
import shutil
import numpy as np
import os
import skimage
import torchvision.transforms as T

from tqdm import tqdm
import argparse

OUR_MAGNIFICATION = 14
OUR_PIXEL_SIZE = 4.25

def align_scale(micro_magn, micro_pixel_size):
    scale_factor = (OUR_MAGNIFICATION / OUR_PIXEL_SIZE) / (micro_magn / micro_pixel_size)
    return round(scale_factor, 2)
    
    
    
if __name__ == '__main__':
    # scale both input (.tif) and ground truth (.png) files!
    # folder_name = 'scale_aligned_dataset_v2_v3'

    parser = argparse.ArgumentParser(description="Align the scales of images using different microscope.")
    parser.add_argument('-m', '--magnification', required=True, help='Microscope\'s magnification')
    parser.add_argument('-s', '--pixel_size', required=True, help='Microscope\'s pixel size')
    parser.add_argument("-p", "--input_path", required=True, help="Path to input file or directory")
    parser.add_argument("-d", "--destination_path", required=True, help="Path to save output file or directory")
    
    args = parser.parse_args()
    
    magnification = float(args.magnification)
    pixel_size = float(args.pixel_size)
    input_path = str(args.input_path)
    destination_path = str(args.destination_path)

    print(f'Magnification: {magnification}')
    print(f'Pixel Size: {pixel_size}')
    print(f"Input Path: {input_path}")
    print(f"Destination Path: {destination_path} \n")
    
    if not os.path.isdir(destination_path):
        os.makedirs(destination_path)
    
    files = [f for f in os.listdir(input_path) if os.path.isfile(input_path + '/' + f)]
    files = [file for file in files if not file.startswith('.')] # remove unwanted files, should contain both tif & png files
    
    # print(files)
    scale_factor = align_scale(micro_magn=magnification, micro_pixel_size=pixel_size)
    # print(scale_factor)
    for image in tqdm(files):
        fname = f'{input_path}/{image}'
        im = skimage.io.imread(fname)
        if len(im.shape) == 2: # gray scale
            im = skimage.transform.rescale(im, scale_factor)
        elif len(im.shape) == 3: # multichannel
            im = skimage.transform.rescale(im, scale_factor, channel_axis=2) # mnfinder data may need to do permutation
        
        destination_fname = f'{destination_path}/{image}'
        to_PIL = T.ToPILImage()
        im_PIL = to_PIL(im)
        # skimage.io.imsave(destination_fname, im)
        im_PIL.save(destination_fname)