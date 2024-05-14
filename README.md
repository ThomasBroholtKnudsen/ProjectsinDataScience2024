# ProjectsinDataScience2024
Repository for the final assignment in course BSPRDAS1KU (spring 2024)

# Dependencies
pip install numpy\
pip install matplotlib\
pip install skimage\
pip install imageio\
pip install os\
pip install pandas\
pip install sklearn\
pip install pickle

# How to use it
## Step 1: Changes in 01_process_images.py\
Line xx: Provide a path for the metadata file.\
Line xx: Provide a path for the folder where the raw images are stored.\
Line xx: Provide a path for the folder where the binary masks are stored.\
Line xx: The path for the file where the feature scores will be stored is set to features/features.csv.\
Line xx: Change the filename for the binary masks. It is assummed that the filenames for the masks consist of the image_id plus a string, to tell that it is a mask. E.g. "PAT_31_42_680_mask.png", where the filename is image_id + "_mask" + file type.\

## Step 2: Changes in 02_train_classifier.py
Line xx: Provide a path for the metadata file.\
Line xx: The path for the file where the feature scores will be stored is set to features/features.csv.\
Line xx: Provide a path for where the classifier file should be stored.\

## Step 3: Changes in 03_evaluate_classifier.py