#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For each dataset and grouping, identify the best performing validation split.

@author: mschroeder
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from util.datasets import DATASETS, GROUPINGS, SPLITS
import pandas as pd

plt.style.use('seaborn-colorblind')

results_root = "/datapc/ob/mschroeder/Ecotaxa/Results"

for dataset in DATASETS:
    print(dataset)
    for grouping in GROUPINGS:
        print(" {}".format(grouping))
        accuracies = np.zeros(len(SPLITS))
        for split in SPLITS:
            results_fn = "SCN_{}_{}_{}/wp2.csv".format(dataset, grouping, split)
            print("  {}".format(results_fn))
            results_fn = os.path.join(results_root, results_fn)
            
            try:
                results = pd.read_csv(results_fn)
            except FileNotFoundError:
                print("   Not found")
                continue
            
            acc = (100 - results["test_3_mistakes"]) / 100
            
            last_epoch_idx = results["epoch"].argmax()
            #print("   Accuracy in last epoch {:d}: {:.2%}".format(results["epoch"][last_epoch_idx], acc[last_epoch_idx]))
            
            accuracies[int(split)] = acc[last_epoch_idx]
            
        best_split = np.argmax(accuracies)
        print("  Best split: {:d} ({:.2%})".format(best_split, accuracies[best_split]))
        
        src =  "./SCN_{}_{}_{}".format(dataset, grouping, best_split)
        
        dst =  "SCN_{}_{}".format(dataset, grouping)
        print("  {}".format(dst))
        dst = os.path.join(results_root, dst)
        
        #os.symlink(src, dst)