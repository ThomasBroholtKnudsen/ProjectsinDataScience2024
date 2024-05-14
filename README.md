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
## Step 1: Changes in 01_process_images.py
Line 33: Provide a path for the metadata file.\
Line 34: Provide a path for the folder where the raw images are stored.\
Line 35: Provide a path for the folder where the binary masks are stored.\
Line 38: The path for the file where the feature scores will be stored is set to features/features.csv.\
Line 63: Change the filename for the binary masks. It is assummed that the filenames for the masks consist of the image_id plus a string, to tell that it is a mask. E.g. "PAT_31_42_680_mask.png", where the filename is image_id + "_mask" + file type.\

## Step 2: Changes in 02_train_classifier.py
Line 26: Provide a path for the metadata file, currently set to "data\metadata.csv".\
Line 42: The path for the file where the feature scores will be stored is set to "features/features.csv".\
Line 127: Provide a path for where the classifier file should be stored, currently sset to "groupOkapi_classifier.sav".\

## Step 3: Changes in 03_evaluate_classifier.py
Line 28: Provide a path for the image, otherwise you will be prompted to do so in the Terminal.\
Line 30: Provide a path for the mask, otherwise you will be prompted to do so in the Terminal.\

## For the Krippendorffs Alpha kripp_alpha_with_computer_scores.R
If wanting to run again, ensure the following:\
Line 5: csv_file_kripp_alpha.csv is in the same location as this script.\
Line 9: features.csv, the output of the generated feature scores is in the same location as this script, or change the file path.\

