"""
FYP project image processing and feature extraction
"""
#-----------------------------------------------------------------
##################### HOW TO USE THE SCRIPT ######################

# Line 33: Provide a path for the metadata file.
# Line 34: Provide a path for the folder where the raw images are stored.
# Line 35: Provide a path for the folder where the binary masks are stored.
# Line 38: The path for the file where the feature scores will be stored is set to features/features.csv.
# Line 44: Provide the column name containing the image IDs from the metadata-file.
# Line 63: Change the filename for the binary masks. 
# It is assummed that the filenames for the masks consist of the image_id plus a string, to tell that it is a mask. 
# E.g. "PAT_31_42_680_mask.png", where the filename is image_id + "_mask" + file type.
#-----------------------------------------------------------------

# Import of necessary libraries and functions
import os
from os.path import exists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our own file that has the feature extraction functions
from extract_features import extract_features

#-------------------
# Main script
#-------------------

#Where is the raw data

file_data = 'data' + os.sep +'okapi_data_metadata.csv'
path_image = 'data' + os.sep + 'Okapi_raw_images'
path_mask = 'data' + os.sep + 'groupOkapi_masks'
  
# Where we will store the features
file_features = 'features/features.csv'

# Read meta-data into a Pandas dataframe
df = pd.read_csv(file_data)

# Extract image IDs and labels from the data. 
image_id = list(df['image_file_name'])

# Getting the number of images. Used later to define size of array to store feature scores. 
num_images = len(image_id)

# Initialize array to store features
feature_names = ['image_id','asymmetry','colour','dots']
num_features = len(feature_names)
features = np.zeros([num_images,num_features], dtype=object)  

#Loop through all images (image_IDs)
for i in np.arange(num_images):
    
    # Define filenames related to this image
    file_image = path_image + os.sep + image_id[i]

    # Define corresponding mask to image. 
    # Split the image_id into name and extension and modify to create filename for mask
    name, extension = os.path.splitext(image_id[i])
    mask_id = f"{name}_mask{extension}"
    file_mask = path_mask + os.sep + mask_id

    # Check if both image and corresponding mask exist, so extract_features can be run
    if exists(file_image) & exists(file_mask):

        # Read the image and mask
        im = plt.imread(file_image)
        im = np.float16(im)
        mask = plt.imread(file_mask)
        mask = np.float16(mask)

        # Measure features
        x = extract_features(im,mask)
           
        # Store image_id and feature scores in the array created earlier
        features[i,0] = image_id[i]
        features[i,1:] = x
        
# Save the feature scores and image_ids to a file by creating a new dataframe with the array of feature scores.
df_features = pd.DataFrame(features, columns=feature_names)     
df_features.to_csv(file_features, index=False)  
