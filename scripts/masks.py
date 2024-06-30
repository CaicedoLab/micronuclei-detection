import numpy as np
import skimage
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from multiprocessing import Pool

if len(sys.argv) < 2:
    print("Use: masks.py plate_dir")
    sys.exit()

path = sys.argv[1]
print(f"Processing {path}")

if os.path.isdir(os.path.join(path, "output")):
    os.system(f"mv {path}/output {path}/prob-maps/")
os.makedirs(os.path.join(path, "masks"), exist_ok=True)


def find_objects(m):
    filepath = os.path.join(path, "prob-maps", m)
    data = np.load(filepath)
    masks = skimage.morphology.label(data > 0.5)
    output_file = filepath.replace("prob-maps","masks").replace(".npy","_masks.tif")
    skimage.io.imsave(output_file, masks)
    print(output_file)


files = [x for x in os.listdir(f"{path}/prob-maps") if x.endswith(".npy")]
with Pool() as p:
    p.map(find_objects, files)

