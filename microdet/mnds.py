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
    otl = read_image(directory, imid, 'phenotype_outlines.png', scale=scale_factor)
    img = read_image(directory, imid, 'phenotype.tif', scale=scale_factor)
    
    # Transform annotations to labels
    otl = otl[:,:,0] > 0 # Use only the red channel
    mask = scipy.ndimage.binary_fill_holes(otl) ^ otl
    labels = skimage.measure.label(mask)
    labels = skimage.morphology.dilation(labels) #recover object edge

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

def read_micronuclei_masks(directory, imid, scale_factor=1.0):
    otl = read_image(directory, imid, 'phenotype_outlines.png', scale=scale_factor)
    edge = otl[:,:,0] > 0 # Use only the red channel
    mask = scipy.ndimage.binary_fill_holes(edge) ^ edge
    return mask + edge

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
    
    def __init__(self, filelist, directory, mode="random", scale_factor=1.0, patch_size=256, stride=8, feature_size=384, edges=False, transform=None):
        # Store parameters
        self.patch_size = patch_size
        self.stride = stride
        self.feature_size = feature_size
        self.mode = mode # in [random, fixed]
        self.edges = edges
        self.transform = transform
        self.shuffled = 0
        
        # Load images and annotations
        all_locs = []
        self.images = {}
        for fname in tqdm(filelist):
            imid = fname.split('.')[0]
            im = read_image(directory, imid, 'phenotype.tif', scale_factor)
            im = np.array((im - np.min(im))/(np.max(im) - np.min(im)), dtype="float32")
            #im = skimage.exposure.rescale_intensity(im, out_range=np.float32)
            mni = read_micronuclei_annotations(directory, imid)
            mnm = read_micronuclei_masks(directory, imid)
            nuc = read_nuclei_masks(directory, imid)
            all_locs.append(mni)
            self.images[imid] = {"image":im, "micro":mnm, "nuclei":nuc, "loc":mni}
            
        self.all_locs = pd.concat(all_locs)
        
        # Validate image sizes
        S = np.asarray([self.images[imid]["image"].shape for imid in self.images.keys()])
        assert np.all(S[:,0] == S[0,0]) and np.all(S[:,1] == S[0,1])
        self.H, self.W = S[0,0], S[0,1]
        
        # Remove excess pixels (alternatively, rescale the images to fit the desired size?)
        self.margin = self.W%self.patch_size
        self.W = self.W - self.margin
        self.H = self.H - self.margin
        
        # Prepare data locations
        if self.mode == "random":
            self.randomize_patch_index()
        elif self.mode == "fixed":
            self.index_patches()
            self.transform = None
        else:
            assert False, "Incorrect mode"

        
    def randomize_patch_index(self):
        self.shuffled += 1
        #print("Randomized",self.shuffled,"times")
        self.index = []
        PS = self.patch_size
        patches_per_image = (self.W // self.patch_size) * (self.H // self.patch_size)
        
        for imid in self.images:
            # Generate random patch coordinates C
            X = np.random.randint(0, self.W - PS, patches_per_image)
            Y = np.random.randint(0, self.H - PS, patches_per_image)
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
                elif (r.y + PS < self.H and r.x + PS < self.W):
                    # If not covered, add a new patch that covers the location
                    extra = [[np.random.randint(max(r.y - PS,0), r.y), np.random.randint(max(r.x - PS,0), r.x)]]
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
        patches_per_image = (self.W // self.patch_size) * (self.H // self.patch_size)
        
        for imid in self.images:
            # Generate regular grid of patch coordinates C
            X = np.linspace(0, self.W - self.W % self.patch_size, self.W // self.patch_size + 1)
            Y = np.linspace(0, self.H - self.H % self.patch_size, self.H // self.patch_size + 1)
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
        
        # Crop patches out of the full image
        PS = self.patch_size
        r,c = int(item["coord"][0]), int(item["coord"][1])
        crop = self.images[item["Image"]]["image"][r:r+PS,c:c+PS]
        mn_mask = self.images[item["Image"]]["micro"][r:r+PS,c:c+PS]
        n_mask = self.images[item["Image"]]["nuclei"][r:r+PS,c:c+PS]
        crop = patch_to_rgb(crop, self.edges)
        mask = torch.Tensor(np.concatenate(
            (mn_mask[np.newaxis,:,:], n_mask[np.newaxis,:,:]), axis=0
        ))
        
        # Move labels to a local reference frame
        #labels = [(min((p[0] - y)//8,31) ,min((p[1] - x)//8,31)) for p in item["locs"]]
        #grid = np.zeros((32,32))
        #for c in labels:
        #    grid[c[0],c[1]] = 1.0
        
        # Apply augmentations
        if self.mode == "random" and self.transform is not None:
            #grid = patch_to_rgb(grid, edges=False)
            #crop, grid = self.transform(crop, grid)
            #grid = grid[0,:,:]
            
            # mask = patch_to_rgb(mask, edges=False)
            crop, mask = self.transform(crop, mask)
            # mask = mask[0,:,:]
            
        return crop, mask
            
        