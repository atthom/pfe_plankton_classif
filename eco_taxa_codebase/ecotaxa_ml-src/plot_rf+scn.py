#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the influence of the various RF parameters on accuracy
(for each dataset and grouping).

Operates on the results of RandomForest training on ZooProcess and SCN features.

@author: mschroeder
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import numpy_indexed as npi
from util.datasets import DATASETS, GROUPINGS, SPLITS
import pandas as pd

plt.style.use('seaborn-colorblind')

results_root = "/datapc/ob/mschroeder/Ecotaxa/Results"

parameters = ['min_samples_leaf', 'n_components', 'n_estimators']
dependent_vars = ['score_train', 'score_val', 'time_fit_train',
       'time_predict_train', 'time_predict_val']

for dataset in DATASETS:
    for grouping in GROUPINGS:
        merged_results = []
        
        fig, axs = plt.subplots(ncols=len(parameters), figsize=(20, 10), sharey=True)
        
        fig.suptitle("{} {}".format(dataset, grouping))
        
        for split in SPLITS:
            results_fn = os.path.join(results_root, "RF+SCN_{}_{}_{}/results.csv".format(dataset, grouping, split))
            try:
                results = pd.read_csv(results_fn)
            except FileNotFoundError:
                continue
            
            merged_results.append(results.assign(split=pd.Series([split]*len(results))))
            
        if len(merged_results) is 0:
            continue
            
        print(dataset, grouping)
        merged_results = pd.concat(merged_results)
        
        print(merged_results.columns)
        
        group_splits = merged_results.groupby(parameters)
        
        split_mean = group_splits.mean().reset_index()
        split_std = group_splits.std().reset_index()
            
        for j, parameter in enumerate(parameters):
            other_parameters = [p for p in parameters if p is not parameter]
            mean_groups = split_mean.groupby(other_parameters)
#            std_groups = dict(iter(split_std.groupby(other_parameters)))

            ax = axs[j]            
            ax.set_title(parameter)
            ax.set_prop_cycle('color',plt.cm.spectral(np.linspace(0,1,len(mean_groups))))
            ax.set_xlim(np.min(split_mean[parameter]), np.max(split_mean[parameter]))
            ax.set_ylabel("score_val")
            
            for other_values, subset in mean_groups:
                label = ", ".join("{}={}".format(k,v) for k, v in zip(other_parameters, other_values))
                subset.plot(parameter, 'score_val',
                            ax=ax,
#                            yerr = std_groups[other_values],
                            xticks=subset[parameter],
                            label=label)
                
        figure_fname = os.path.join(results_root, "RF+SCN_{}_{}.pdf".format(dataset, grouping))
        fig.savefig(figure_fname, bbox_inches="tight")