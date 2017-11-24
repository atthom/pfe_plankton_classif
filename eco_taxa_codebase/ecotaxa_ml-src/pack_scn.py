#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pack models so that they can be imported in EcoTaxa.

@author: mschroeder
"""

import os
from util.datasets import DATASETS, GROUPINGS
from util.features import load_features2, train_feature_pca
import errno
import shutil
import json

dataset_root = "/datapc/ob-ssd/mschroeder/Ecotaxa/Datasets/"
results_root = "/datapc/ob/mschroeder/Ecotaxa/Results"
model_root = "/data1/mschroeder/Ecotaxa/Models"

# Number is yet to defined
n_components = 50

for dataset in DATASETS:
    for grouping in GROUPINGS:
        model_name = "SCN_{}_{}".format(dataset, grouping)
        print(model_name)
        
        scn_dir = "SCN_{}_{}_TEST".format(dataset, grouping)
        scn_dir = os.path.join(results_root, scn_dir)
        
        # Check existence of scn_dir
        if not os.path.isdir(scn_dir):
            print(" {} does not exist.".format(scn_dir))
            continue
        
        classlist_fn = "{}_classes_{}.txt".format(dataset, grouping)
        classlist_fn = os.path.join(dataset_root, classlist_fn)
        
        weights_fn = os.path.join(scn_dir, "_epoch-99.cnn")
        
        # Check existence of class list, weights, PCA
        if not os.path.isfile(classlist_fn):
            print(" {} does not exist.".format(classlist_fn))
            continue
        
        if not os.path.isfile(weights_fn):
            print(" {} does not exist.".format(weights_fn))
            continue
    
        features_fn = os.path.join(scn_dir, "_train.features")
        
        features = load_features2(features_fn)
        pca = train_feature_pca(features, features_fn, verbose=True, n_components=n_components)
        
        n_objects, n_features = features["X"].shape
        
        pca_fn = os.path.join(scn_dir, "_train.features.pca_{}.jbl".format(n_components))
        
        if not os.path.isfile(pca_fn):
            print(" {} does not exist. This is an error!".format(pca_fn))
            continue
            
        model_dir = os.path.join(model_root, model_name)
        
        try:
            os.mkdir(model_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print("{} exists. Skipping.".format(model_dir))
                continue
            raise
        
        # Copy data over
        shutil.copy(classlist_fn, os.path.join(model_dir, "classes.txt"))
        shutil.copy(weights_fn, model_dir)
        shutil.copy(pca_fn, os.path.join(model_dir, "feature_pca.jbl"))
        
        metadata = {
            "name": model_name,
            "type": "SparseConvNet",
            "epoch": 99,
            "n_objects": n_objects,
            "n_features": n_features
        }
        
        with open(os.path.join(model_dir, "meta.json"), "w") as f:
            json.dump(metadata, f, indent=4);
        