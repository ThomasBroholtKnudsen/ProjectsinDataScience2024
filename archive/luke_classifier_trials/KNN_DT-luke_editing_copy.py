import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


# Load data
file_data = '..' + os.sep + 'data' + os.sep + 'okapi_data_metadata.csv'
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
annotation_data = '..' + os.sep + 'features' + os.sep + 'features_okapi_images.csv'
our_annotations = pd.read_csv(annotation_data)
features = our_annotations[['asymmetry', 'colour', 'dots']].values

# Split data for training and testing (final evaluation)
X_train, X_test, y_train, y_test = train_test_split(features, cancer, test_size=0.2, random_state=1907, stratify=cancer)

# Print data shapes to illustrate split amounts
print(f"Total features shape: {features.shape}")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Training labels:\n{y_train}")
print(f"Testing labels:\n{y_test}")

# Decision Tree Model
dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train, y_train)
y_pred_tree = dec_tree.predict(X_test)
balanced_accuracy_tree = balanced_accuracy_score(y_test, y_pred_tree)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print("Decision Tree Test Accuracy:", accuracy_tree)

# Plot the Decision Tree
plt.figure(figsize=(10, 8))
plot_tree(dec_tree, filled=True, feature_names=['asymmetry', 'colour', 'dots'], class_names=['Non-cancerous', 'Cancerous'])
plt.title("Decision Tree Model")
#plt.show()

# K-Nearest Neighbors Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
balanced_accuracy_knn = balanced_accuracy_score(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Test Accuracy:", accuracy_knn)

# Output results for both models
print(f"Decision Tree Accuracy: {accuracy_tree:.4f}")
print(f"KNN Accuracy: {accuracy_knn:.4f}")

num_folds = 5
num_classifiers = 2
kfold = KFold(n_splits = num_folds)

acc_val = np.empty([num_folds,num_classifiers])

for i_data, (train_index, val_index) in enumerate(kfold.split(X_train,y_train)):
    #the number of training data sets created will therefore also be according to the num_folds
    x_variant_train = X_train[train_index,:]
    y_variant_train = y_train[train_index]
    x_variant_val = X_train[val_index,:]
    y_variant_val = y_train[val_index]

    dec_tree.fit(x_variant_train, y_variant_train)
    y_tree_prediction = dec_tree.predict(x_variant_val)
    acc_val[i_data,0] = balanced_accuracy_score(y_variant_val, y_tree_prediction)
    print(f"This iteration of decision tree score is: {acc_val[i_data,0]}")

    knn.fit(x_variant_train, y_variant_train)
    y_knn_prediction = knn.predict(x_variant_val)
    acc_val[i_data,1] = balanced_accuracy_score(y_variant_val, y_knn_prediction)
    print(f"This iteration of knn tree score is: {acc_val[i_data,1]}")

print(acc_val)
    


