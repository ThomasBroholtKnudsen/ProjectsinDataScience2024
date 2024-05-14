#-----------------------------------------------------------------
##################### HOW TO USE THE SCRIPT ######################

# Line 26: Provide a path for the metadata file, currently set to "data\metadata.csv".
# Line 42: The path for the file where the feature scores will be stored is set to "features/features.csv".
# Line 127: Provide a path for where the classifier file should be stored, currently sset to "groupOkapi_classifier.sav".

#-----------------------------------------------------------------


# Import of necessary libraries and functions
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

import pickle #for saving/loading trained classifiers

random_state = 42

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
annotation_data = 'features' + os.sep + 'features.csv'
our_annotations = pd.read_csv(annotation_data)
features = our_annotations[['asymmetry', 'colour', 'dots']].values

# Split data for training and testing (final evaluation)
X_train, X_test, y_train, y_test = train_test_split(features, cancer, test_size=0.2, stratify=cancer, random_state = 42)

#Kfolding, and running on all models
num_folds = 10
num_classifiers = 5
kfold = KFold(n_splits = num_folds)

acc_val = np.empty([num_folds,num_classifiers])

#this will run a for loop on each set of training/validation data
for i_data, (train_index, val_index) in enumerate(kfold.split(X_train,y_train)):
    #the number of training data sets created will therefore also be according to the num_folds
    x_variant_train = X_train[train_index,:]
    y_variant_train = y_train[train_index]
    x_variant_val = X_train[val_index,:]
    y_variant_val = y_train[val_index]

    dec_tree = DecisionTreeClassifier(random_state = 42)
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

    forest = RandomForestClassifier()
    forest.fit(x_variant_train, y_variant_train)
    y_pred_forest = forest.predict(x_variant_val)
    bal_acc_forest = balanced_accuracy_score(y_variant_val, y_pred_forest)
    acc_val[i_data,3] = bal_acc_forest

    #hyper parameters used are commonly found ones, used by https://www.youtube.com/watch?v=_QuGM_FW9eo&list=PLcQVY5V2UY4LNmObS0gqNVyNdVfXnHwu8&index=9
    param_forest = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', min_samples_split = 10, max_depth = 7, random_state=42)
    param_forest.fit(x_variant_train, y_variant_train)
    y_pred_paramforest = param_forest.predict(x_variant_val)
    bal_acc_paramforest = balanced_accuracy_score(y_variant_val, y_pred_paramforest)
    acc_val[i_data,4] = bal_acc_paramforest

#Average over all folds for each classifier
average_acc = np.mean(acc_val,axis=0) 

print("The best classifier based on validation is:")
best_acc_score = max(average_acc)
if best_acc_score == average_acc[0]:
    classifier = DecisionTreeClassifier(random_state = 42)
    print("DT")
elif best_acc_score == average_acc[1]:
    classifier = KNeighborsClassifier(1)
    print("KNN1")
elif best_acc_score == average_acc[2]:
    classifier = KNeighborsClassifier(5)
    print("KNN5")
elif best_acc_score == average_acc[3]:
    classifier = RandomForestClassifier(random_state = 42)
    print("RF")
else:
    classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', min_samples_split = 10, max_depth = 7, random_state=42)
    print("ParamRF")
print(f"It's average balanced accuracy score is: {best_acc_score}")

#testing this best classifier on our set aside test data
classifier.fit(X_train, y_train)
y_best_pred = classifier.predict(X_test)
bal_acc_score_best = balanced_accuracy_score(y_test, y_best_pred)
print(f'Our best classifier on the validation data has a balanced accuracy score of: {bal_acc_score_best} when tested on the set aside test data')


#training this model on our entire dataset
classifier = classifier.fit(features, cancer)

#This is the classifier you need to save using pickle, add this to your zip file submission
filename = 'groupOkapi_classifier.sav'
pickle.dump(classifier, open(filename, 'wb'))



