#-----------------------------------------------------------------
##################### HOW TO USE THE SCRIPT ######################

#Line 28: Provide a path for the image, otherwise you will be prompted to do so in the Terminal.
#Line 30: Provide a path for the mask, otherwise you will be prompted to do so in the Terminal.

#-----------------------------------------------------------------

import pickle #for loading your trained classifier

from extract_features import extract_features #our feature extraction

#For reading in the image, mask, and metadata
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Import our own file that has the feature extraction functions
from extract_features import extract_features

#for testing the classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#put the relative file locations for the image and mask here
print("Please type in the relative file path of the image")
file_image = str(input())
print("Please type in the relative file path of the mask")
file_mask = str(input())

#read in the image and mask into arrays
im_plt = plt.imread(file_image)
img = np.float16(im_plt)
mask_plt = plt.imread(file_mask)
mask = np.float16(mask_plt)

# The function that should classify new images. 
# The image and mask are the same size, and are already loaded using plt.imread
def classify(img, mask):
    
     #Extract features (the same ones that you used for training)
     x = extract_features(img, mask)
     y = x.reshape(1,-1)
     
     #Load the trained classifier
     classifier = pickle.load(open('groupOkapi_classifier.sav', 'rb'))
    
    
     #Use it on this example to predict the label AND posterior probability
     pred_label = classifier.predict(y)
     pred_prob = classifier.predict_proba(y)

     #Since our saved classifier is KNN1, this will always return 1 or 0
     if pred_label == 1:
          print("Expected class is: Cancer")
          print(f'Probability of prediction is: {int(pred_prob[0,1])}')
     else:
          print("Expected class is: Not cancer")
          print(f'Probability of prediction is: {int(pred_prob[0,0])}')

     
     
     #print('predicted label is ', pred_label)
     #print('predicted probability is ', pred_prob)
     return pred_label, pred_prob
 
    
# The TAs will call the function above in a loop, for external test images/masks
classify(img,mask)