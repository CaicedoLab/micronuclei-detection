import zipfile
import os
from skimage import io, measure
import matplotlib.pyplot as plt
import numpy as np

# Define the path to the zip file
zip_path = "/scr/data/micronuclei-screen/prob-maps/plate_1/masks.zip"
extraction_path = "/scr/data/micronuclei-screen/prob-maps/plate_1"
extracted_path = "/scr/data/micronuclei-screen/prob-maps/plate_1/masks"


# Unzip the file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

# List all image files in the extracted folder
image_files = [f for f in os.listdir(extracted_path) if f.endswith(('.png', '.tif', '.jpg'))]


# List to hold the sizes of all objects
object_sizes = []

# Iterate over each image file
for image_file in image_files:
    print("processing", image_file)
    # Load the image
    image_path = os.path.join(extracted_path, image_file)
    image = io.imread(image_path)

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
plt.title('Distribution of Object Sizes')
plt.xlabel('Object Size (pixels)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
