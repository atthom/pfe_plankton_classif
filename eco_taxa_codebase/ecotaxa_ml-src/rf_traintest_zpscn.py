#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Random Forest on the ZooProcess and SCN features of the full training set
and evaluate on the testing set.

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
from util.features import load_features2 as load_features
from util.features import train_feature_pca
from util.join_recarrays import join_2d_arrays
from util.labels import LabelEncoder
from util.zooscan import load_datafile

DATASET = os.environ["DATASET"]
GROUPING  = os.environ["GROUPING"]
RESULTS_DIR = os.environ["RESULTS_DIR"]
DATASET_ROOT = os.environ["DATASET_ROOT"]
SCN_DIR = os.environ["SCN_DIR"]

N_COMPONENTS = 50

classes_fn = os.path.join(DATASET_ROOT, DATASET + "_classes_" + GROUPING + ".txt")

scn_train_fn = os.path.join(SCN_DIR, "_train.features")
scn_test_fn = os.path.join(SCN_DIR, "_test.features")

zooprocess_data_fn = os.path.join(DATASET_ROOT, DATASET + "_data.csv")

results_fn = os.path.join(RESULTS_DIR, "results.csv")

print("Arguments:")
for arg in "DATASET,GROUPING,RESULTS_DIR,DATASET_ROOT,data_fn,classes_fn,results_fn".split(","):
    print(" {}: {}".format(arg, globals().get(arg)))
print()

X_fields = ZOOPROCESS_FIELDS[DATASET].split(",")
y_field = "name_" + GROUPING

print("Loading data...")
time_start = time.time()
zooprocess_data = load_datafile(zooprocess_data_fn)
scn_train_data = load_features(scn_train_fn)
scn_test_data = load_features(scn_test_fn)
print("Done. (%.2f seconds)" % (time.time() - time_start))

print("Zooprocess data contains fields:", ",".join(zooprocess_data.dtype.names))
print("Retained fields:", ",".join(X_fields))

zooprocess_X = get_feature_array(zooprocess_data, X_fields)

# Load or fit PCA
pca = train_feature_pca(scn_train_data, data_filename = scn_train_fn, n_components=N_COMPONENTS)

print("Transforming (Keeping {:d} components)...".format(N_COMPONENTS))
scn_train_X = pca.transform(scn_train_data["X"])
scn_test_X = pca.transform(scn_test_data["X"])

zooprocess_n_features = zooprocess_X.shape[1]
scn_n_features = scn_train_X.shape[1]

print("Number of ZooProcess features: {:d}".format(zooprocess_n_features))
print("Number of SCN features: {:d}".format(scn_n_features))

print("Joining ZooProcess and SCN features...")
train_X, idx_zp, idx_scn = join_2d_arrays(zooprocess_data["objid"], scn_train_data["objid"], zooprocess_X, scn_train_X, return_indices=True)
#train_y = zooprocess_data[y_field][idx_zp]
#assert np.all(zooprocess_data["objid"][idx_zp] == )
#assert np.all(train_y == scn_train_data["y"][idx_scn])
train_y = scn_train_data["y"][idx_scn]

test_X, idx_zp, idx_scn = join_2d_arrays(zooprocess_data["objid"], scn_test_data["objid"], zooprocess_X, scn_test_X, return_indices=True)
#test_y = zooprocess_data[y_field][idx_zp]
#assert np.all(test_y == )
test_y = scn_test_data["y"][idx_scn]

print("Training set (dtype/shape):", train_X.dtype, train_X.shape, train_y.dtype, train_y.shape)
print("Testing set (dtype/shape):", test_X.dtype, test_X.shape, test_y.dtype, test_y.shape)

# Encode labels
classes = np.loadtxt(classes_fn, dtype="U32", delimiter=",", usecols=0)
le = LabelEncoder(classes)

print("Using {:,d} / {:,d} (training / validation) objects in {:d} classes.".format(len(train_y), len(test_y), len(le.encoded_labels())))

# Parameter grid
# Ecotaxa currently uses n_estimators=300, min_samples_leaf=5, min_samples_split=10
parameters = {
    'min_samples_leaf': [2], # [2, 5, 10],
    'n_estimators': [10, 100, 300],
    # 'max_features': [4, 7, 11],
    }

parameter_names = list(parameters.keys())

param_n_estimators = parameters.pop('n_estimators')
parameter_grid = ParameterGrid(parameters)

print("Results will be written to {}.".format(results_fn))
with open(results_fn, "w", 1) as f:
    fieldnames = ["model_id"] + parameter_names \
        + ["time_fit_train", "time_predict_train", "score_train", "time_predict_test", "score_test"]
    writer = csv.DictWriter(f, fieldnames, delimiter=",")
    writer.writeheader()
    
    model_counter = itertools.count()
    for params in parameter_grid:
        base_result = params.copy()
        
        print(", ".join(("%s: %s" % kv) for kv in base_result.items()))
        
        model_fname = "random_forest_{}.jbl".format("_".join([str(v) for k,v in sorted(params.items())]))
        model_fname = os.path.join(RESULTS_DIR, model_fname)
        
        # Initialize model
        model = RandomForestClassifier(class_weight = "balanced", warm_start=True, n_jobs=6, **params)
        
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
                model.fit(train_X, train_y)
            total_time_fit_train += time.perf_counter() - start
            result["time_fit_train"] = total_time_fit_train
            
            # Training set score
            start = time.perf_counter()
            y_pred_train = model.predict(train_X)
            time_taken = time.perf_counter() - start
            result["time_predict_train"] = time_taken
            result["score_train"] = accuracy_score(train_y, y_pred_train)
            
            # Validation set score
            start = time.perf_counter()
            y_pred_test = model.predict(test_X)
            time_taken = time.perf_counter() - start
            result["time_predict_test"] = time_taken
            result["score_test"] = accuracy_score(test_y, y_pred_test)
            
            conf_mat = confusion_matrix(test_y, y_pred_test, labels=le.encoded_labels())
            
            conf_mat_fname = os.path.join(RESULTS_DIR, "confusion_matrix_{:d}.csv".format(model_id))
            np.savetxt(conf_mat_fname, conf_mat)
            
            print(" {} training, training score {:.2%}, testing score {:.2%}.".format(
                    humanize.naturaldelta(result["time_fit_train"]),
                    result["score_train"],
                    result["score_test"]))
            
            writer.writerow(result)

            print("  Saving model to {}.".format(model_fname))
            try:
                with open(model_fname, "wb") as f:
                    joblib.dump(model, model_fname)
            except MemoryError as e:
                print("  Model could not be saved:", e)
