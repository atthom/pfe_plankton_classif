#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Random Forest on the SCN features of the full training set and evaluate
on the testing set.

@author: mschroeder
"""

import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from util.features import load_features2 as load_features
from util.features import train_feature_pca

SCN_DIR = os.environ["SCN_DIR"]
RESULTS_DIR = os.environ["RESULTS_DIR"]

N_COMPONENTS = 50

scn_train_fn = os.path.join(SCN_DIR, "_train.features")
scn_test_fn = os.path.join(SCN_DIR, "_test.features")
results_fn = os.path.join(RESULTS_DIR, "results.csv")

print("Loading data...")
start = time.perf_counter()
scn_train_data = load_features(scn_train_fn)
scn_test_data = load_features(scn_test_fn)
time_taken = time.perf_counter() - start
print("Done ({:f}s).".format(time_taken))

# Load or fit PCA
pca = train_feature_pca(scn_train_data, data_filename = scn_train_fn, n_components=N_COMPONENTS)

X_train = pca.transform(scn_train_data["X"])
X_test = pca.transform(scn_test_data["X"])

y_train = scn_train_data["y"]
y_test = scn_test_data["y"]

# Parameter grid
# Ecotaxa currently uses n_estimators=300, min_samples_leaf=5, min_samples_split=10
parameters = {
        'min_samples_leaf': 2,
        'n_estimators': 300
}

# Initialize model
model = RandomForestClassifier(class_weight = "balanced",
                               n_jobs=4,
                               **parameters)

result = parameters.copy()

model_fname = os.path.join(RESULTS_DIR, "rf_model.pickle")

# Train
print("Training...")
start = time.perf_counter()
model.fit(X_train, y_train)
result["time_fit_train"] = time.perf_counter() - start

# Training set score
start = time.perf_counter()
y_pred_train = model.predict(X_train)
time_taken = time.perf_counter() - start
result["time_predict_train"] = time_taken
result["score_train"] = accuracy_score(y_train, y_pred_train)

# Validation set score
start = time.perf_counter()
y_pred_test = model.predict(X_test)
time_taken = time.perf_counter() - start
result["time_predict_test"] = time_taken
result["score_test"] = accuracy_score(y_test, y_pred_test)

conf_mat = confusion_matrix(y_test, y_pred_test)

conf_mat_fname = os.path.join(RESULTS_DIR, "confusion_matrix.csv")
np.savetxt(conf_mat_fname, conf_mat)

print("# Results")
for k, v in sorted(result.items()):
    print("{}: {}".format(k, v))
print("---")

print("Saving model to {}.".format(model_fname))
try:
    with open(model_fname, "wb") as f:
        joblib.dump(model, model_fname)
except Exception as e:
    print("Model could not be saved:", e)
