import os
import skimage
import random
import torch
import numpy as np
import pandas as pd
import scipy

import skimage.morphology

from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# GRAY SCALE PATCH TO RGB IMAGE
def patch_to_rgb(patch, edges=False):
    if edges:
        sobel = skimage.filters.sobel(patch)
        sobel = 2*skimage.exposure.rescale_intensity(sobel, out_range=np.float32)
        sobel[sobel > 1.] = 1.
    else:
        sobel = patch
        
    px = np.concatenate(
        (sobel[np.newaxis,:,:], patch[np.newaxis,:,:], patch[np.newaxis,:,:]), 
        axis=0)
    return torch.Tensor(px)

# OPEN AN IMAGE
def read_image(directory, imid, suffix, scale=1.0):
    imname = f'{directory}/{imid}.{suffix}'
    im = skimage.io.imread(imname)
    if scale != 1.:
        im = skimage.transform.rescale(im, scale)
    return im


# READ MICRONUCLEI ANNOTATIONS
def read_micronuclei_annotations(directory, imid, size_filter=1e9, scale_factor=1.0):
    mask = read_image(directory, imid, 'phenotype_outlines.png', scale=scale_factor)
    img = read_image(directory, imid, 'phenotype.tif', scale=scale_factor)
    
    labels = skimage.measure.label(mask)
    data = []
    for i in range(1,len(np.unique(labels))):
        ys,xs = np.where(labels == i)
        intensity = np.mean(img[ys,xs])
        a,b = int(np.mean(ys)), int(np.mean(xs))
        area = np.sum(labels == i)
        if area <= size_filter:
            a = int(scale_factor * a)
            b = int(scale_factor * b)
            data.append({"Image":f'{directory}/{imid}.phenotype.tif', "x":b, "y":a, "area":area, "intensity": intensity})

    mni = pd.DataFrame(data=data, columns=["Image","x","y","area","intensity"])
    return mni #, labels


def read_nuclei_masks(directory, imid, scale_factor=1.0):
    # otl = read_image(directory, imid, 'nuclei.tif', scale=scale_factor)
    otl = read_image(directory, imid, 'nuclei-clean.tif', scale=scale_factor)
    otl = otl > 0 # It's a labeled matrix, so make it binary
    return otl

# PATCH AUGMENTATIONS
def detection_transforms(patch, target):
    # Rotations
    if random.random() > 0.25:
        angle = random.choice([90, 180, 270])
        patch = TF.rotate(patch, angle)
        target = TF.rotate(target, angle)
    
    # Horizontal flips
    if random.random() > 0.5:
        patch = TF.hflip(patch)
        target = TF.hflip(target)
    
    # Brightness adjustments
    if random.random() > 0.5:
        brightness = min(2., max(0.5, np.random.normal(1, 0.5)))
        patch = TF.adjust_brightness(patch, brightness)
       
    # Contrast adjustments
    if random.random() > 0.5:
        contrast = min(2., max(0.5, np.random.normal(1, 0.5)))
        patch = TF.adjust_contrast(patch, contrast)
    
    return patch, target

# DATASET CLASS
class MicronucleiDataset(Dataset):
    
    def __init__(self, filelist, directory, mode="random", scale_factor=1.0, patch_size=256, stride=8, feature_size=384, edges=False, transform=None, gaussian=False):
        # Store parameters
        self.patch_size = patch_size
        self.stride = stride
        self.feature_size = feature_size
        self.mode = mode # in [random, fixed]
        self.edges = edges
        self.transform = transform
        self.shuffled = 0
        self.gaussian = gaussian
        self.directory = directory
        
        # Load images and annotations
        # all_locs = []
        self.images = {}
        for fname in tqdm(filelist):
            imid = fname.split('.')[0]
            im = read_image(directory, imid, 'phenotype.tif', scale_factor)
            im = np.array((im - np.min(im))/(np.max(im) - np.min(im)), dtype="float32")
            #im = skimage.exposure.rescale_intensity(im, out_range=np.float32)
            mni = read_micronuclei_annotations(directory, imid)
            mnm = read_image(directory, imid, 'phenotype_outlines.png', scale_factor)
            nuc = read_nuclei_masks(directory, imid)
            # all_locs.append(mni)
            self.images[imid] = {"image":im, "micro":mnm, "nuclei":nuc, "loc":mni}
            
        # self.all_locs = pd.concat(all_locs)
        
        # Prepare data locations
        if self.mode == "random":
            self.randomize_patch_index()
        elif self.mode == "fixed":
            self.index_patches()
            self.transform = None
        else:
            assert False, "Incorrect mode"

        
    def randomize_patch_index(self): # This function is only applied to training set
        self.shuffled += 1
        #print("Randomized",self.shuffled,"times")
        self.index = []
        PS = self.patch_size
        
        df = pd.read_csv(self.directory.replace('train', 'metadata.csv'))
        for imid in self.images:
            # Generate random patch coordinates C
            H, W = self.images[imid]["image"].shape
            patches_per_image = (W // PS) * (H // PS)
            
            # Oversample more crops
            subset = df.loc[df.filenames == imid + '.phenotype.tif', 'datasets'].iloc[0]
            if subset in ['HeLa', 'RPE1']:
                patches_per_image = patches_per_image * 7
            elif subset in ['BBBC039', 'mnfinder_train']:
                patches_per_image = patches_per_image * 5
                
            # print(f'{imid}: height - {H}, width - {W}, patches - {patches_per_image}')
            X = np.random.randint(0, W - PS, patches_per_image)
            Y = np.random.randint(0, H - PS, patches_per_image)
            C = np.stack((Y,X)).T
            A = {}

            # Micronuclei locations
            for k,r in self.images[imid]["loc"].iterrows():
                # Check whether the location r.x,r.y is covered by patches
                matches = np.where(np.logical_and( 
                            np.logical_and(C[:,0] < r.y, C[:,0] + PS > r.y),
                            np.logical_and(C[:,1] < r.x, C[:,1] + PS > r.x)
                ))
                matches = matches[0]
                if len(matches) > 0:
                    # Annotate all patches that cover the location
                    for m in matches:
                        try: A[m].append((r.y, r.x))
                        except: A[m] = [(r.y, r.x)]
                elif (r.y + PS < H or r.x + PS < W): # or statement might fit non-square images?
                    # If not covered, add a new patch that covers the location
                    if (r.y == 0 and r.x == 0):
                        extra = [[0, 0]]
                    elif (r.y != 0 and r.x == 0):
                        if (r.y + PS > H):
                            extra = [[H - PS, 0]]
                        else:
                            extra = [[np.random.randint(max(r.y - PS,0), r.y), 0]]
                    elif (r.y == 0 and r.x != 0):
                        if (r.x + PS > W):
                            extra = [[0, W - PS]]
                        else:
                            extra = [[0, np.random.randint(max(r.x - PS,0), r.x)]]
                    else:
                        if (r.y + PS < H and r.x + PS < W):
                            extra = [[np.random.randint(max(r.y - PS,0), r.y), np.random.randint(max(r.x - PS,0), r.x)]]
                        elif (r.y + PS < H and r.x + PS > W):
                            extra = [[np.random.randint(max(r.y - PS,0), r.y), W - PS]]
                        elif (r.y + PS > H and r.x + PS < W):
                            extra = [[H - PS, np.random.randint(max(r.x - PS,0), r.x)]]
                        elif (r.y + PS > H and r.x + PS > W):
                            extra = [[H - PS, W - PS]]
                    C = np.append(C, extra, axis=0)
                    A[C.shape[0]-1] = [(r.y, r.x)]

            # Put annotated patches in the index
            count = 0
            for k in A:
                self.index.append({"Image":imid, "coord": C[k,:].tolist(), "locs":A[k]})
                count += 1

            # Complete budget with non-annotated patches
            U = [x for x in range(patches_per_image) if x not in A]
            pointer = 0
            while count < patches_per_image:
                k = U[pointer]
                self.index.append({"Image":imid, "coord": C[k,:].tolist(), "locs":[]})
                pointer += 1
                count += 1
                

    def index_patches(self):
        self.index = []
        PS = self.patch_size
        
        for imid in self.images:
            # Generate regular grid of patch coordinates C
            H, W = self.images[imid]["image"].shape
            patches_per_image = (W // PS) * (H // PS)
            # print(f'{imid}: height - {H}, width - {W}, patches - {patches_per_image}')
            X = np.linspace(0, W - W % PS, W // PS + 1)
            Y = np.linspace(0, H - H % PS, H // PS + 1)
            X,Y = np.meshgrid(X[:-1],Y[:-1], indexing='ij')
            X = X.reshape((patches_per_image,))
            Y = Y.reshape((patches_per_image,))
            C = np.stack((Y,X)).T
            A = {}

            # Micronuclei locations
            for k,r in self.images[imid]["loc"].iterrows():
                # Find which patches cover the location r.x,r.y
                matches = np.where(np.logical_and( 
                            np.logical_and(C[:,0] < r.y, C[:,0] + PS > r.y),
                            np.logical_and(C[:,1] < r.x, C[:,1] + PS > r.x)
                ))
                matches = matches[0]
                if len(matches) > 0:
                    # Annotate all patches that cover the location
                    for m in matches:
                        try: A[m].append((r.y, r.x))
                        except: A[m] = [(r.y, r.x)]
                
                # comment ouf only for grid search purpose, remove commenting after grid search
                # else:
                #     print(f"{imid}: Micronuclei at ({r.y},{r.x}) is not covered by any patches")

            # Put all patches in the index
            for k in range(C.shape[0]):
                try:
                    self.index.append({"Image":imid, "coord": C[k,:].tolist(), "locs":A[k]})
                except:
                    self.index.append({"Image":imid, "coord": C[k,:].tolist(), "locs":[]})
                              
        
    def __len__(self):
        return len(self.index)
        
        
    def __getitem__(self, idx):
        item = self.index[idx]
        # print(item['Image'])
        
        if self.gaussian: # default: False
            height, width = np.random.normal(loc=self.patch_size, scale=20, size=2) # 20 is the best so far
            height, width = int(height), int(width)
            r,c = int(item["coord"][0]), int(item["coord"][1])
            crop = self.images[item["Image"]]["image"][r:r+height,c:c+width]
            mn_mask = self.images[item["Image"]]["micro"][r:r+height,c:c+width]
            n_mask = self.images[item["Image"]]["nuclei"][r:r+height,c:c+width]
            
            crop = patch_to_rgb(crop, self.edges)
            mask = torch.Tensor(np.concatenate(
                (mn_mask[np.newaxis,:,:], n_mask[np.newaxis,:,:]), axis=0
            ))

            # interpolate to Patch Size x Patch Size before concatention
            # interpolation messed up binary values sometime, remember to check
            crop = torch.nn.functional.interpolate(crop.unsqueeze(0), (self.patch_size, self.patch_size))
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0), (self.patch_size, self.patch_size))
            
            crop = crop.squeeze(0)
            mask = mask.squeeze(0)
            
            # print(f'Crop shape: {crop.shape}')
            # print(f'Mask shape: {mask.shape}')
        
        # Crop patches out of the full image
        else: # original code
            PS = self.patch_size
            r,c = int(item["coord"][0]), int(item["coord"][1])
            crop = self.images[item["Image"]]["image"][r:r+PS,c:c+PS]
            mn_mask = self.images[item["Image"]]["micro"][r:r+PS,c:c+PS]
            n_mask = self.images[item["Image"]]["nuclei"][r:r+PS,c:c+PS]
            crop = patch_to_rgb(crop, self.edges)
            mask = torch.Tensor(np.concatenate(
                (mn_mask[np.newaxis,:,:], n_mask[np.newaxis,:,:]), axis=0
            ))
        
        if self.mode in ["random"] and self.transform is not None:
            crop, mask = self.transform(crop, mask)
            # mask = mask[0,:,:]
            
        return crop, mask
            
        