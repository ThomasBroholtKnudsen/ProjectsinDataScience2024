
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

import pickle #for saving/loading trained classifiers


# Load data
file_data = 'data' + os.sep + 'okapi_data_metadata.csv'
df = pd.read_csv(file_data)

# Prepare labels
label = np.array(df['diagnosis'])
cancer = np.copy(label)
cancer[cancer == "BCC"] = 1
cancer[cancer == "MEL"] = 1
cancer[cancer == "SCC"] = 1
cancer[cancer == "ACK"] = 0
cancer[cancer == "NEV"] = 0
cancer[cancer == "SEK"] = 0
cancer = cancer.astype(int)  # Ensure that labels are integers


# Load features
annotation_data = 'features' + os.sep + 'features_okapi_images.csv'
our_annotations = pd.read_csv(annotation_data)
features = our_annotations[['asymmetry', 'colour', 'dots']].values

# Split data for training and testing (final evaluation)
X_train, X_test, y_train, y_test = train_test_split(features, cancer, test_size=0.2, stratify=cancer)

#Kfolding, and running on all models
num_folds = 10
num_classifiers = 3
kfold = KFold(n_splits = num_folds)

acc_val = np.empty([num_folds,num_classifiers])

for i_data, (train_index, val_index) in enumerate(kfold.split(X_train,y_train)):
    #the number of training data sets created will therefore also be according to the num_folds
    x_variant_train = X_train[train_index,:]
    y_variant_train = y_train[train_index]
    x_variant_val = X_train[val_index,:]
    y_variant_val = y_train[val_index]

    dec_tree = DecisionTreeClassifier()
    dec_tree.fit(x_variant_train, y_variant_train)
    y_tree_prediction = dec_tree.predict(x_variant_val)
    acc_val[i_data,0] = balanced_accuracy_score(y_variant_val, y_tree_prediction)

    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn1.fit(x_variant_train, y_variant_train)
    y_pred_knn1 = knn1.predict(x_variant_val)
    bal_acc_knn1 = balanced_accuracy_score(y_variant_val, y_pred_knn1)
    acc_val[i_data,1] = bal_acc_knn1

    knn5 = KNeighborsClassifier(n_neighbors=5)
    knn5.fit(x_variant_train, y_variant_train)
    y_pred_knn5 = knn5.predict(x_variant_val)
    bal_acc_knn5 = balanced_accuracy_score(y_variant_val, y_pred_knn5)
    acc_val[i_data,2] = bal_acc_knn5

#Average over all folds
average_acc = np.mean(acc_val,axis=0) 

#The best classifier is.....
if average_acc[0] > average_acc[1] and average_acc[0] > average_acc[2]:
    classifier = DecisionTreeClassifier()
elif average_acc[1] > average_acc[0] and average_acc[1] > average_acc[2]:
    classifier = KNeighborsClassifier(1)
else:
    classifier = KNeighborsClassifier(5)


#training this model on our entire dataset
classifier = classifier.fit(features, cancer)

#This is the classifier you need to save using pickle, add this to your zip file submission
filename = 'groupOkapi_classifier.sav'
pickle.dump(classifier, open(filename, 'wb'))



