#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Random Forest on the ZooProcess features of the full training set
and evaluate on the testing set.

@author: mschroeder
"""

import os
import time
import warnings

import humanize
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from util.datasets import ZOOPROCESS_FIELDS
from util.labels import LabelEncoder
from util.zooscan import load_datafile

# max_features:
#   Increasing max_features generally improves the performance of the model.
#   You decrease the speed of algorithm by increasing the max_features.
# n_estimators:
#   Higher number of trees give you better performance but makes your code slower.
#   You should choose as high value as your processor can handle because this makes your predictions stronger and more stable.
# min_sample_leaf:
#   A smaller leaf makes the model more prone to capturing noise in train data. Prediction error increases with higher numbers.
# min_samples_split (ignore as complementary to min_sample_leaf):
#   The minimum number of samples required to split an internal node. Smaller numbers give better results.
# criterion:
#   Criterion "has little effect on the performance of decision tree induction algorithms." (Tan et al. Introduction to Data Mining 2006)
# max_features:
#   Number of features considered for each split.


# flowcam, uvp5ccelter, zoocam, zooscan
DATASET = os.environ["DATASET"]
GROUPING  = os.environ["GROUPING"]
RESULTS_DIR = os.environ["RESULTS_DIR"]
DATASET_ROOT = os.environ["DATASET_ROOT"]

train_data_fn = os.path.join(DATASET_ROOT, DATASET + "_data_train.csv")
test_data_fn = os.path.join(DATASET_ROOT, DATASET + "_data_test.csv")
classes_fn = os.path.join(DATASET_ROOT, DATASET + "_classes_" + GROUPING + ".txt")
results_fn = os.path.join(RESULTS_DIR, "results.csv")

print("Arguments:")
for arg in "DATASET,GROUPING,RESULTS_DIR,DATASET_ROOT,data_fn,classes_fn,results_fn".split(","):
    print(" {}: {}".format(arg, globals().get(arg)))
print()

# The variables characterising the images are in the _data.csv file. The fields are between area and cdexc. Some are useless for prediction:
# - all the ones that start with x and y (position on the zooscan glass)
# - angle: Angle between the primary axis and a line par- allel to the x-axis of the image
# - bx,by: coordinates of the top left point of the smallest rectangle enclosing the object
# - tag?
# - angle
# - xmg5

X_fields = ZOOPROCESS_FIELDS[DATASET].split(",")
y_field = "name_" + GROUPING

min_samples = 10


print("Loading data...")
time_start = time.time()
train_data = load_datafile(train_data_fn)
test_data = load_datafile(test_data_fn)
print("Done. (%.2f seconds)" % (time.time() - time_start))

#data = data[:1000]

print("Training data contains fields:", ",".join(train_data.dtype.names))
print("Testing data contains fields:", ",".join(train_data.dtype.names))

if sorted(train_data.dtype.names) != sorted(train_data.dtype.names):
    print("Data fields do not match.")
    
def check_variance(data, fields):
    for name in fields:
        try:
            var = np.var(data[name])
            if np.allclose(var, 0):
                print("Variance is 0 for field {}.".format(name))
        except TypeError:
            print("Variance could not be calculated for {}.".format(name))

# Report fields of variance=0
check_variance(train_data, X_fields)
check_variance(test_data, X_fields)

def remove_empty(data):
    selector = data[y_field] != ""
    print("{} empty entries for target in {} are removed.".format(np.sum(~selector), y_field))
    return data[selector]

# Remove entries with empty class
train_data = remove_empty(train_data)
test_data = remove_empty(test_data)

def recode_simple(data, fields):
    # Recode as simple array
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return data[fields].view("float32").reshape(data.shape[0], -1)

X_train = recode_simple(train_data, X_fields)
X_test = recode_simple(test_data, X_fields)

# Encode labels
classes = np.loadtxt(classes_fn, dtype="U32", delimiter=",", usecols=0)
le = LabelEncoder(classes)
y_train = le.transform(train_data[y_field])
y_test = le.transform(test_data[y_field])

print("Using %d objects in %d classes for training" % (len(X_train), len(classes)))
print("Using %d objects in %d classes for testing" % (len(X_test), len(classes)))

# TODO: Detect NaN values
#contains_NaN = False
#for name in X_fields:
#    nan_selector = np.isnan(data[name])
#    if nan_selector.any():
#        print("Field {} contains {} NaN values.".format(name, np.sum(nan_selector)))
#        print(data["objid"][nan_selector])
#        contains_NaN = True
#        
#if contains_NaN:
#    raise ValueError("Data contains NaN values.")
    
    
# Parameter grid
# Ecotaxa currently uses n_estimators=300, min_samples_leaf=5, min_samples_split=10
parameters = {
    'n_estimators': 300,
    'min_samples_leaf': 2,
    'max_features': 11}

# Initialize model
model = RandomForestClassifier(class_weight = "balanced", n_jobs=4, **parameters)

result = parameters.copy()

# Train
print("Training...")
start = time.perf_counter()
model.fit(X_train, y_train)
time_fit_train = time.perf_counter() - start
result["time_fit_train"] = time_fit_train

# Training set score
print("Testing (training set)...")
start = time.perf_counter()
y_pred_train = model.predict(X_train)
time_taken = time.perf_counter() - start
result["time_predict_train"] = time_taken
result["score_train"] = accuracy_score(y_train, y_pred_train)

# Test set score
print("Testing (testing set)...")
start = time.perf_counter()
y_pred_val = model.predict(X_test)
time_taken = time.perf_counter() - start
result["time_predict_test"] = time_taken
result["score_test"] = accuracy_score(y_test, y_pred_val)

conf_mat = confusion_matrix(y_test, y_pred_val, labels=le.encoded_labels())

conf_mat_fname = os.path.join(RESULTS_DIR, "confusion_matrix.csv")
np.savetxt(conf_mat_fname, conf_mat)

print("{} training, training score {:.2%}, test score {:.2%}.".format(
        humanize.naturaldelta(result["time_fit_train"]),
        result["score_train"],
        result["score_test"]))

print("# Results")
for k, v in sorted(result.items()):
    print("{}: {}".format(k, v))
