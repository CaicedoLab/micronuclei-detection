import os
import numpy as np
import skimage
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(
        description="Reformat microsam Data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )

parser.add_argument('--load_path', type=str, help='annotated micronuclei datasets',
                    default='/scr/data/annotated_mn_datasets/')
parser.add_argument('--save_path', type=str, help='path to save cropped images for training microSAM',
                    default='/hdd/jcaicedo/projects/micronuclei_detection/Train_and_Eval/mndino_data/baselines/microsam_data/')

args = parser.parse_args()
load_path = args.load_path
save_path = args.save_path

folders = ['train', 'validation', 'test']

for folder in tqdm(folders):
    # LOAD_PATH = f'/scr/yren/annotated_mn_datasets/{folder}/'
    # SAVE_PATH = f'/scr/yren/microsam_data/{folder}/'
    LOAD_PATH = os.path.join(load_path, folder)
    SAVE_PATH = os.path.join(save_path, folder)
    
    images = os.listdir(os.path.join(LOAD_PATH, 'images'))
    images = [image for image in images if not image.startswith('.')]
    images.sort()
    
    gts = os.listdir(os.path.join(LOAD_PATH, 'mn_masks'))
    gts = [gt for gt in gts if not gt.startswith('.')]
    gts.sort()
    
    PS = 256
    for i in tqdm(range(len(gts))):
        im = skimage.io.imread(os.path.join(LOAD_PATH, 'images', images[i]))
        gt = skimage.io.imread(os.path.join(LOAD_PATH, 'mn_masks', gts[i]))

        assert im.shape == gt.shape
        H,W = im.shape
        patches_per_image = (W // PS) * (H // PS)
        X = np.linspace(0, W - W % PS, W // PS + 1)
        Y = np.linspace(0, H - H % PS, H // PS + 1)
        X,Y = np.meshgrid(X[:-1],Y[:-1], indexing='ij')
        X = X.reshape((patches_per_image,))
        Y = Y.reshape((patches_per_image,))
        C = np.stack((Y,X)).T
        
        idx = 0
        for j in range(len(C)):
            r,c = C[j]
            r,c = int(r), int(c)
            # check if gt patch has micronuclei or not
            gt_patch = gt[r:r+PS, c:c+PS]
            gt_labels = skimage.morphology.label(gt_patch)
            if np.max(gt_labels) > 0: # 1 represents only background
                # save the corresponding input and gt
                im_patch = im[r:r+PS, c:c+PS]
                
                # reconstruct filename
                imid = images[i].split('.')[0]
                suffix = f'.crop{idx}'
                new_imid = imid + suffix
                
                # im_patch = (im_patch - np.min(im_patch)) / (np.max(im_patch) - np.min(im_patch)) * 255
                skimage.io.imsave(os.path.join(SAVE_PATH, new_imid + '.tif'), im_patch)
                skimage.io.imsave(os.path.join(SAVE_PATH, new_imid + '.png'), gt_labels.astype(np.uint16))
                idx = idx + 1