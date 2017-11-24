#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pack models so that they can be imported in EcoTaxa.

@author: mschroeder
"""

import os
from util.datasets import DATASETS, GROUPINGS, ZOOPROCESS_FIELDS
from util.features import load_features2
import errno
import shutil
import json
from util.zooscan import load_datafile
import numpy as np

dataset_root = "/datapc/ob-ssd/mschroeder/Ecotaxa/Datasets/"
results_root = "/datapc/ob/mschroeder/Ecotaxa/Results"
model_root = "/data1/mschroeder/Ecotaxa/Models"

for dataset in DATASETS:
    for grouping in GROUPINGS:
        model_name = "RF_ZPSCN_{}_{}".format(dataset, grouping)
        print(model_name)
        
        results_dir = "RF_ZP+SCN_{}_{}_TEST".format(dataset, grouping)
        results_dir = os.path.join(results_root, results_dir)
        
        scn_dir = "SCN_{}_{}_TEST".format(dataset, grouping)
        scn_dir = os.path.join(results_root, scn_dir)
        
        scn_model_name = "SCN_{}_{}".format(dataset, grouping)
        
        # Check existence of results_dir
        if not os.path.isdir(results_dir):
            print(" {} does not exist.".format(results_dir))
            continue
        
        classlist_fn = "{}_classes_{}.txt".format(dataset, grouping)
        classlist_fn = os.path.join(dataset_root, classlist_fn)
        rf_model_fn = os.path.join(results_dir, "random_forest_2.jbl")
        features_fn = os.path.join(scn_dir, "_train.features")
        zooprocess_data_fn = os.path.join(dataset_root, dataset + "_data.csv")
        
        # Check existence of source files
        if not os.path.isfile(classlist_fn):
            print(" {} does not exist.".format(classlist_fn))
            continue
        
        if not os.path.isfile(rf_model_fn):
            print(" {} does not exist.".format(rf_model_fn))
            continue
        
        model_dir = os.path.join(model_root, model_name)
        
        try:
            os.mkdir(model_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print("{} exists. Skipping.".format(model_dir))
                continue
            raise
        
        print("Loading SCN features to get their number...")
        features = load_features2(features_fn)
        n_objects = len(features["X"])
        
        print("Loading ZooProcess features to calculate features medians...")
        zooprocess_data = load_datafile(zooprocess_data_fn)
        
        zooprocess_fields = ZOOPROCESS_FIELDS[dataset].split(",")
        
        # Calculate median of every field
        zp_medians = [float(np.median(zooprocess_data[field])) for field in zooprocess_fields]
        
        # Copy data over
        shutil.copy(classlist_fn, os.path.join(model_dir, "classes.txt"))
        shutil.copy(rf_model_fn, os.path.join(model_dir, "random_forest.jbl"))
        
        metadata = {
            "name": model_name,
            "type": "RandomForest-ZooProcess+SparseConvNet",
            "scn_model": scn_model_name,
            # Number of objects used in the training of the model
            "n_objects": n_objects,
            # ZooProcess features used (in order)
            "zooprocess_fields:": zooprocess_fields,
            "zooprocess_medians": zp_medians
        }
        
        with open(os.path.join(model_dir, "meta.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        