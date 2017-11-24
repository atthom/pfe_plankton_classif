#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Random Forest on the ZooProcess features of all training splits
and evaluate on the corresponding validation splits.

@author: mschroeder
"""

import csv
import itertools
import os
import time
import warnings

import humanize
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from util.datasets import ZOOPROCESS_FIELDS
from util.feature_array import get_feature_array
from util.labels import LabelEncoder
from util.zooscan import load_datafile

# Currently, ecotaxa uses the following configuration:
# RandomForestClassifier(n_estimators=300, min_samples_leaf=5, min_samples_split=10)

# max_features:
#   Increasing max_features generally improves the performance of the model.
#   You decrease the speed of algorithm by increasing the max_features.
# n_estimators:
#   Higher number of trees give you better performance but makes your code slower.
#   You should choose as high value as your processor can handle because this makes your predictions stronger and more stable.
# min_sample_leaf:
#   A smaller leaf makes the model more prone to capturing noise in train data. Prediction error increases with higher numbers.
#   Usually 5. 2: Deeper trees -> more overfitting, 10: Less overfitting
# min_samples_split (ignore as complementary to min_sample_leaf):
#   The minimum number of samples required to split an internal node. Smaller numbers give better results.
# criterion:
#   Criterion "has little effect on the performance of decision tree induction algorithms." (Tan et al. Introduction to Data Mining 2006)
# max_features:
#   Number of features considered for each split. sqrt=7 is usual. 4, 7, 11.


# flowcam, uvp5ccelter, zoocam, zooscan
DATASET = os.environ["DATASET"]
GROUPING  = os.environ["GROUPING"]
RESULTS_DIR = os.environ["RESULTS_DIR"]
DATASET_ROOT = os.environ["DATASET_ROOT"]


data_fn = os.path.join(DATASET_ROOT, DATASET + "_data_train.csv")
base = os.path.splitext(os.path.basename(data_fn))[0]
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
data = load_datafile(data_fn)
print("Done. (%.2f seconds)" % (time.time() - time_start))

#data = data[:1000]

print("Data contains fields:", ",".join(data.dtype.names))

# Report fields of variance=0
for name in X_fields:
    try:
        var = np.var(data[name])
        if np.allclose(var, 0):
            print("Variance is 0 for field {}.".format(name))
    except TypeError:
        print("Variance could not be calculated for {}.".format(name))

# Remove entries with empty class
selector = data[y_field] != ""
print("{} empty entries for target in {} are removed.".format(np.sum(~selector), y_field))
data = data[selector]

# Recode as simple array
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore", FutureWarning)
#    X = data[X_fields].view("float32").reshape(data.shape[0], -1)
    
X = get_feature_array(data, X_fields)

# Encode labels
classes = np.loadtxt(classes_fn, dtype="U32", delimiter=",", usecols=0)
le = LabelEncoder(classes)
y = le.transform(data[y_field])

## Calculate class weights
#class_weights = len(y) / (len(classes) * np.bincount(y))
#print("Class weights:", *class_weights)

print("Using %d objects in %d classes." % (len(X), len(classes)))

split_ids = data["split"]

# Detect NaN values
contains_NaN = False
for name in X_fields:
    nan_selector = np.isnan(data[name])
    if nan_selector.any():
        print("Field {} contains {} NaN values.".format(name, np.sum(nan_selector)))
        print(data["objid"][nan_selector])
        contains_NaN = True
        
if contains_NaN:
    raise ValueError("Data contains NaN values.")
    
    
# Parameter grid
# Ecotaxa currently uses n_estimators=300, min_samples_leaf=5, min_samples_split=10
parameters = {
    'n_estimators': [10, 100, 300, 400],
    'min_samples_leaf': [2, 5, 10],
    'max_features': [4, 7, 11],
    'split': [0,1,2]}

parameter_names = list(parameters.keys())

param_n_estimators = parameters.pop('n_estimators')
parameter_grid = ParameterGrid(parameters)

print("Results will be written to {}.".format(results_fn))
with open(results_fn, "w", 1) as f:
    fieldnames = ["model_id"] + parameter_names \
        + ["time_fit_train", "time_predict_train", "score_train", "time_predict_val", "score_val"]
    writer = csv.DictWriter(f, fieldnames, delimiter=",")
    writer.writeheader()
    
    model_counter = itertools.count()
    for params in parameter_grid:
        base_result = params.copy()
        
        # Don't pass split as parameter to RandomForestClassifier 
        split = params.pop('split')
        
        print(", ".join(("%s: %s" % kv) for kv in base_result.items()))
        
        model_fname = "random_forest_{}.jbl".format("_".join([str(v) for k,v in sorted(params.items())]))
        model_fname = os.path.join(RESULTS_DIR, model_fname)
        
        validation_selector = split_ids % 3 == split
        training_selector = ~validation_selector
        
        X_train = X[training_selector]
        y_train = y[training_selector]
        X_val = X[validation_selector]
        y_val = y[validation_selector]
        
        # Initialize model
        model = RandomForestClassifier(class_weight = "balanced", warm_start=True, n_jobs=4, **params)
        
        total_time_fit_train = 0
        
        for n_estimators in sorted(param_n_estimators):
            model_id = next(model_counter)
            result = base_result.copy()
            result["model_id"] = model_id
            result["n_estimators"] = n_estimators
            
            print(" Model {:d}, n_estimators: {:d}".format(model_id, n_estimators))
            
            model.n_estimators = n_estimators
        
            # Train
            start = time.perf_counter()
            with warnings.catch_warnings():
                # Ignore the following UserWarning:
                # class_weight presets "balanced" or "balanced_subsample" are not recommended for warm_start if the fitted data differs from the full dataset. In order to use "balanced" weights, use compute_class_weight("balanced", classes, y). In place of y you can use a large enough sample of the full training set target to properly estimate the class frequency distributions. Pass the resulting weights as the class_weight parameter.
                warnings.filterwarnings("ignore", "class_weight.*not recommended for warm_start.*", UserWarning)
                model.fit(X_train, y_train)
            total_time_fit_train += time.perf_counter() - start
            result["time_fit_train"] = total_time_fit_train
            
            # Training set score
            start = time.perf_counter()
            y_pred_train = model.predict(X_train)
            time_taken = time.perf_counter() - start
            result["time_predict_train"] = time_taken
            result["score_train"] = accuracy_score(y_train, y_pred_train)
            
            # Validation set score
            start = time.perf_counter()
            y_pred_val = model.predict(X_val)
            time_taken = time.perf_counter() - start
            result["time_predict_val"] = time_taken
            result["score_val"] = accuracy_score(y_val, y_pred_val)
            
            conf_mat = confusion_matrix(y_val, y_pred_val, labels=le.encoded_labels())
            
            conf_mat_fname = os.path.join(RESULTS_DIR, "confusion_matrix_{:d}.csv".format(model_id))
            np.savetxt(conf_mat_fname, conf_mat)
            
            print(" {} training, training score {:.2%}, validation score {:.2%}.".format(
                    humanize.naturaldelta(result["time_fit_train"]),
                    result["score_train"],
                    result["score_val"]))
            
            writer.writerow(result)
            
            print("  Saving model to {}.".format(model_fname))
            try:
                with open(model_fname, "wb") as f:
                    joblib.dump(model, model_fname)
            except MemoryError as e:
                print("  Model could not be saved:", e)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#ax.scatter(
#    results["param_n_estimators"],
#    results["param_min_samples_split"],
#    results['mean_test_score'])
#ax.set_xlabel("param_n_estimators")
#ax.set_ylabel("param_min_samples_split")
#ax.set_zlabel("mean accuracy")
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#ax.scatter(
#    results["param_n_estimators"],
#    results["param_min_samples_split"],
#    results['mean_fit_time'])
#ax.set_xlabel("param_n_estimators")
#ax.set_ylabel("param_min_samples_split")
#ax.set_zlabel("mean_fit_time")
#
# mean_fit_time increases with param_n_estimators
# With higher param_min_samples_split, mean_fit_time increases more slowly.
#
# for p in parameters.keys():
#    plt.figure()
#    plt.title(p)
#    plt.plot(results["param_" + p], results['mean_test_score'])
#    plt.xlabel(p)
#    plt.ylabel("mean_test_score")
