### Reformat mnfinders data into our pipeline, including reformat nucleus mask from png to tiff
import os
import skimage # using skimage is better
from tqdm import tqdm

'''
The name is repeated, keep the folder name in front of it.
'''

# resize the image here! (including both train and testing)

train_path = './training'
os.makedirs('./mnfinder_training', exist_ok=True)
for folder in tqdm(os.listdir(train_path)):
    # prevent ./DS_Store in MacOS system
    if not '.DS_Store' in folder:
        for phenotype in os.listdir(f'{train_path}/{folder}/images/'):
            # print(phenotype)
            filename = phenotype.split('.')[0]
            filename = filename.replace(' ', '_')
            source = f'{train_path}/{folder}/images'
            destination = f'./mnfinder_training/{folder}-{filename}.phenotype.tif'

            im = skimage.io.imread(f'{source}/{phenotype}')
            skimage.io.imsave(destination, im)
            
        for mn_mask in os.listdir(f'{train_path}/{folder}/mn_masks'):
            filename = mn_mask.split('.')[0]
            filename = filename.replace(' ', '_')
            source = f'{train_path}/{folder}/mn_masks'
            destination = f'./mnfinder_training/{folder}-{filename}.phenotype_outlines.png'
            
            im = skimage.io.imread(f'{source}/{mn_mask}')
            skimage.io.imsave(destination, im)
            
        for n_mask in os.listdir(f'{train_path}/{folder}/nucleus_masks'):
            # convert png to tiff
            filename = n_mask.split('.')[0]
            filename = filename.replace(' ', '_')
            source = f'{train_path}/{folder}/nucleus_masks/'
            destination = f'./mnfinder_training/{folder}-{filename}.nuclei.tif'

            im = skimage.io.imread(f'{source}/{n_mask}')
            skimage.io.imsave(destination, im)
            


test_path = './testing'
os.makedirs('./mnfinder_testing', exist_ok=True)
for phenotype in os.listdir(f'{test_path}/images/'):
    filename = phenotype.split('.')[0]
    filename = filename.replace(' ', '_')
    source = f'{test_path}/images'
    destination = f'./mnfinder_testing/{filename}.phenotype.tif'

    im = skimage.io.imread(f'{source}/{phenotype}')
    skimage.io.imsave(destination, im)

for mn_mask in os.listdir(f'{test_path}/mn_masks'):
    filename = mn_mask.split('.')[0]
    filename = filename.replace(' ', '_')
    source = f'{test_path}/mn_masks'
    destination = f'./mnfinder_testing/{filename}.phenotype_outlines.png'

    im = skimage.io.imread(f'{source}/{mn_mask}')
    skimage.io.imsave(destination, im)
