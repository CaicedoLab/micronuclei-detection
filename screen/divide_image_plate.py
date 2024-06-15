import os
import sys



plate_folder_path = ""
store_subset_path = "/staging/groups/caicedo_group/micronuclei/screen-plates-divided"
import os
import zipfile
import math

def divide_files_into_zips(source_dir, dest_dir, base_zip_name, num_zips=8):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Get all files in the source directory
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    # Calculate the number of files per zip
    num_files_per_zip = math.ceil(len(files) / num_zips)
    
    for i in range(num_zips):
        # Create the zip file name
        zip_name = f"{base_zip_name}_{i+1}.zip"
        zip_path = os.path.join(dest_dir, zip_name)
        
        # Create a zip file
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add files to the zip file
            for j in range(i * num_files_per_zip, min((i + 1) * num_files_per_zip, len(files))):
                file_path = os.path.join(source_dir, files[j])
                zipf.write(file_path, os.path.basename(file_path))
        print(f"Created {zip_name} in {dest_dir}")

# Example usage
base_zip_name = "plate_1_images"
divide_files_into_zips(plate_folder_path, store_subset_path, base_zip_name)
