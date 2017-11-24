#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Random Forest on the SCN features of a training split and evaluate
on the validation split.

@author: mschroeder

Q: Does PCA impact RF classification?
A: No.
    Training/prediction time is unaffected by dimensionality reduction.
    Accuracy is largely unaffected. Reductions to as few as 10 dimensions are tolerable.

model.feature_importances_ contains the relative importance of each feature and sums up to 1.

"""

import csv
import itertools
import os
import pickle
import time
import warnings

import humanize
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from util.features import load_features2 as load_features
from util.trim_estimators import trim_pca

SCN_DIR = os.environ["SCN_DIR"]
RESULTS_DIR = os.environ["RESULTS_DIR"]

train_fn = os.path.join(SCN_DIR, "_train.features")
val_fn = os.path.join(SCN_DIR, "_val.features")
results_fn = os.path.join(RESULTS_DIR, "results.csv")

print("Loading data...")
start = time.perf_counter()
train_data = load_features(train_fn)
val_data = load_features(val_fn)
time_taken = time.perf_counter() - start
print("Done ({:f}s).".format(time_taken))

# Load or fit PCA
pca_model_fname = os.path.join(RESULTS_DIR, "pca.pickle")

if os.path.isfile(pca_model_fname):
    print("Loading PCA...")
    
    with open(pca_model_fname, "rb") as f:
        pca = pickle.load(f)
else:
    print("Fitting PCA...")
    pca = PCA()
    start = time.perf_counter()
    pca.fit(train_data["X"])
    time_taken = time.perf_counter() - start
    print("Done ({:f}s).".format(time_taken))

    print(" Saving PCA model to {}.".format(pca_model_fname))
    with open(pca_model_fname, "wb") as f:
        pickle.dump(pca, f, protocol=pickle.HIGHEST_PROTOCOL)

def broken_stick(n_components):
    """
    https://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/en_Tanagra_Nb_Components_PCA.pdf
    """
    result = np.zeros(n_components)
    for k in range(n_components):
        result[k] = sum(1 / i for i in range(k+1, n_components+1))
        
    return result

print("Determining the right number of components in PCA:")
# Broken stick
keep_bs = np.argmax(pca.explained_variance_ < broken_stick(pca.n_components_))
print("Broken stick: {:d} components".format(keep_bs + 1))
# Threshold: mean eigenvalue
keep_me = np.argmax(pca.explained_variance_ < np.mean(pca.explained_variance_))
print("Threshold of mean eigenvalue: {:d} components".format(keep_me + 1))
# Proportion of variance explained (95%)
cum_exv = np.cumsum(pca.explained_variance_ratio_)
keep_pv = np.argmax(cum_exv > .95)
print("95% variance explained: {:d} components".format(keep_pv + 1))


# Parameter grid
# Ecotaxa currently uses n_estimators=300, min_samples_leaf=5, min_samples_split=10
parameters = {
        "n_components": [0, 112, 30], #[0, 448, 112, 30],
        'min_samples_leaf': [2, 5],
}

param_n_estimators = [10, 100, 300, 400]

parameter_grid = ParameterGrid(parameters)

print("Performing grid search on {:d} combinations of parameters...".format(len(param_n_estimators) * len(parameter_grid)))

print("Results will be written to {}.".format(results_fn))
with open(results_fn, "w", 1) as f:
    fieldnames = ["model_id"] + list(parameters.keys()) \
        + ["n_estimators", "time_fit_train", "time_predict_train", "score_train", "time_predict_val", "score_val"]
    writer = csv.DictWriter(f, fieldnames, delimiter=",")
    writer.writeheader()
    
    model_counter = itertools.count()
    
    for params in parameter_grid:
        base_result = params.copy()
        
        print(", ".join(("%s: %s" % kv) for kv in base_result.items()))
        
        model_fname = "model_{}.pickle".format("_".join([str(v) for k,v in sorted(params.items())]))
        model_fname = os.path.join(RESULTS_DIR, model_fname)
        
        n_components = params.pop("n_components")
        
        if n_components == 0:
            X_train = train_data["X"]
            X_val = val_data["X"]
        else:
            pca_trunc = trim_pca(pca, n_components)
            X_train = pca_trunc.transform(train_data["X"])
            X_val = pca_trunc.transform(val_data["X"])
            
        #print(" Shapes of training and validation data:", X_train.shape, X_val.shape)
            
        y_train = train_data["y"]
        y_val = val_data["y"]
        
        
        # Initialize model
        # TODO: Load existing model
        model = RandomForestClassifier(class_weight = "balanced",
                                       warm_start=True,
                                       n_jobs=4,
                                       **params)
        
        total_time_fit_train = 0
        
        for n_estimators in sorted(param_n_estimators):
            model_id = next(model_counter)
            
            result = base_result.copy()
            result["model_id"] = model_id
            result["n_estimators"] = n_estimators
            
            print(" Model {:d}, n_estimators: {:d}".format(model_id, n_estimators))
            
            model.n_estimators = n_estimators
            
            # Train
            print("  Training...")
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
            
            conf_mat = confusion_matrix(y_val, y_pred_val)
            
            conf_mat_fname = os.path.join(RESULTS_DIR, "confusion_matrix_{:d}.csv".format(model_id))
            np.savetxt(conf_mat_fname, conf_mat)
            
            print("  {} training, training score {:.2%}, validation score {:.2%}.".format(
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
