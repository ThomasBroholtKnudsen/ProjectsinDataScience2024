

import pickle #for loading your trained classifier

from extract_features import extract_features #our feature extraction

#For reading in the image, mask, and metadata
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

#for testing the classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#put the relative file locations for the image and mask here
file_image = '..'+ os.sep +'data' + os.sep + 'PAT_123_456_789.png'
file_mask = '..'+ os.sep +'data' + os.sep + 'PAT_123_456_789_mask.png'

#read in the image and mask into arrays
im_plt = plt.imread(file_image)
img = np.float16(im_plt)
mask_plt = plt.imread(file_mask)
mask = np.float16(mask_plt)

#relative file location for metadata (groundtruth)
file_data = '..'+ os.sep +'data' + os.sep + 'metadata.csv'
ground_truth = pd.read_csv(file_data)


# The function that should classify new images. 
# The image and mask are the same size, and are already loaded using plt.imread
def classify(img, mask):
    
     #Extract features (the same ones that you used for training)
     x = extract_features(img, mask)
         
     
     #Load the trained classifier
     classifier = pickle.load(open('groupOkapi_classifier.sav', 'rb'))
    
    
     #Use it on this example to predict the label AND posterior probability
     pred_label = classifier.predict(x)
     pred_prob = classifier.predict_proba(x)
     
     
     #print('predicted label is ', pred_label)
     #print('predicted probability is ', pred_prob)
     return pred_label, pred_prob
 
    
# The TAs will call the function above in a loop, for external test images/masks