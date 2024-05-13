import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Read in metadata in a dataframe
df = pd.read_csv('okapi_data_metadata.csv')

# Turning the diagnosis binary, 1 = cancer, 0 = not cancer
label = df['diagnosis'].map({"BCC" : 1, "MEL" : 1, "SCC" : 1, "ACK" : 0, "NEV" : 0, "SEK" : 0}) 

# Read in feature scores as a dataframe
our_annotations = pd.read_csv('features_okapi_images.csv')

merged_df = pd.concat([our_annotations, label], axis=1)

# Drop the first column, it should not affect the model
train_df = merged_df.drop(columns = ['image_id'])

X = train_df.iloc[:, 0:3]

y = train_df.iloc[:, 3]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.2)

rf = RandomForestClassifier()

# Fit model
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(rf.score(X_test, y_test))

print(classification_report(y_test, y_pred))

features = pd.DataFrame(rf.feature_importances_, index=X.columns)

print(features.head(15))

# With hyper parameters adjusted
rf2 = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', min_samples_split = 10, max_depth = 14, random_state = 42)

# Fit 2nd model
rf2.fit(X_train, y_train)

print(rf2.score(X_test, y_test))

y_pred2 = rf2.predict(X_test)

print(classification_report(y_test, y_pred2))

features = pd.DataFrame(rf2.feature_importances_, index=X.columns)

print(features.head(15))
