import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print("Decision Tree Test Accuracy:", accuracy_tree)

# Plot the Decision Tree
plt.figure(figsize=(10, 8))
plot_tree(dec_tree, filled=True, feature_names=['asymmetry', 'colour', 'dots'], class_names=['Non-cancerous', 'Cancerous'])
plt.title("Decision Tree Model")
plt.show()

# K-Nearest Neighbors Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Test Accuracy:", accuracy_knn)

# Output results for both models
print(f"Decision Tree Accuracy: {accuracy_tree:.4f}")
print(f"KNN Accuracy: {accuracy_knn:.4f}")
