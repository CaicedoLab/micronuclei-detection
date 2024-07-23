import os
import numpy as np
import zipfile
import os
from skimage import io, measure
import matplotlib.pyplot as plt
import numpy as np

for i in range(3, 9):
    # Define the path to the zip file
    zip_path = f"/scr/data/micronuclei-screen/prob-maps/plate_{i}/prob-maps.zip"
    extraction_path = f"/scr/data/micronuclei-screen/prob-maps/plate_{i}"
    extracted_path = f"/scr/data/micronuclei-screen/prob-maps/plate_{i}/prob-maps"


    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)

    # List all image files in the extracted folder
    image_files = [f for f in os.listdir(extracted_path) if f.endswith(('.png', '.tif', '.jpg', '.npy'))]


    # List to hold the sizes of all objects
    object_sizes = []

    # Iterate over each image file
    for image_file in image_files:
        print("processing", image_file)
        # Load the image
        image_path = os.path.join(extracted_path, image_file)
        image = np.load(image_path, allow_pickle=True)

        # Label the image
        label_image = measure.label(image)
        
        # Measure properties of labeled regions
        properties = measure.regionprops(label_image)
        
        # Extract the size of each object
        for prop in properties:
            object_sizes.append(prop.area)

    # Create a histogram of object sizes
    plt.figure(figsize=(10, 6))
    plt.hist(object_sizes, bins=50, color='blue', edgecolor='black')
    plt.title(f"Distribution of object sizes, Number of objects: {len(object_sizes)}, Number of Images: {len(image_files)}")
    plt.xlabel('Object Size (pixels)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f"/home/MORGRIDGE/vagrawal/vidit/num_objects/object_sizes_plate_{i}.png")
    plt.show()

    #Remove the unzipped folder
    os.system(f"rm -r {extracted_path}")
    print("remove folder: ", extracted_path)