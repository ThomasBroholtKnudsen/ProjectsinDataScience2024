"""
FYP project imaging
"""

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
#### TK - following is original code. Doesn't work for me. Problem with os.sep. EDIT: Works as of 08/05.
file_data = 'data' + os.sep +'metadata.csv'
path_image = 'data' + os.sep + 'images' # + os.sep + 'imgs_part_1' # Left out depending on structure of data
path_mask = 'data' + os.sep + 'masks'

# Where is the raw data
#### TK - following is my attempt at fixing above code. EDIT: Above works. 
#file_data = os.path.join('data', 'metadata.csv')
#path_image = os.path.join('data', 'images') #,'imgs_part_1') # Left out depending on structure of data
#path_mask = os.path.join('data','masks')

  
# Where we will store the features
file_features = 'features/features.csv'


#Read meta-data into a Pandas dataframe
df = pd.read_csv(file_data)

# Extract image IDs and labels from the data. 
image_id = list(df['img_id'])
label = np.array(df['diagnostic'])

# We don't use the following as of 07/05-24
# Here you could decide to filter the data in some way (see task 0)
# For example you can have a file selected_images.csv which stores the IDs of the files you need
#is_nevus =  label == 'NEV'

# Getting the number of images. Used later to define size of array to store feature scores. 
num_images = len(image_id)


#Make array to store features
feature_names = ['asymmetry','colour','dots']
num_features = len(feature_names)
features = np.zeros([num_images,num_features], dtype=np.float16)  
#TK test to use list of lists instead of numpy arrays:
#features = [ [0, 0, 0] for i in range(num_images)]

#Loop through all images
for i in np.arange(num_images):
    
    # Define filenames related to this image
    #### TK - following line works even though I had trouble earlier with os.sep
    file_image = path_image + os.sep + image_id[i]

    # Define corresponding mask to image
    # Split the image_id into name and extension
    name, extension = os.path.splitext(image_id[i])
    mask_id = f"{name}_mask{extension}"
    file_mask = path_mask + os.sep + mask_id



    if exists(file_image) & exists(file_mask):
        # Read the image
        im = plt.imread(file_image)
        im = np.float16(im)
        mask = plt.imread(file_mask)
        mask = np.float16(mask)
    
        # Measure features.
        x = extract_features(im,mask)
        
        #### TK - test to include image_id as a column in the features array. Doesn't work, as features is of data type float. 
        #x = np.concatenate((np.array([image_id[i]]), extract_features(im)))
        
           
        # Store in the array we created before
        features[i,:] = x
        #TK - use if we create list of lists instead of using numpy arrays
        #features[i] = x
       
        
#Save the image_id used + features to a file by creating a new dataframe with the array of feature scores. 
#TK - use if we create list of lists instead of np array:
#features_array = np.array(features)
df_features = pd.DataFrame(features, columns=feature_names)     
df_features.to_csv(file_features, index=False)  
