#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the influence of the various RF parameters on accuracy
(for each dataset and grouping).

Operates on the results of RandomForest training on ZooProcess features.

@author: mschroeder
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from util.datasets import DATASETS, GROUPINGS
import pandas as pd

plt.style.use('seaborn-colorblind')

results_root = "/datapc/ob/mschroeder/Ecotaxa/Results"

parameters = ['n_estimators', 'min_samples_leaf', 'max_features']
dependent_vars = ['score_val'] # 'time_predict_val', 'time_fit_train', 'time_predict_train', 'score_train',

for dataset in DATASETS:
    for grouping in GROUPINGS:
        results_fn = os.path.join(results_root, "RF_{}_{}/results.csv".format(dataset, grouping))
        try:
            results = pd.read_csv(results_fn)
        except FileNotFoundError:
            print("{} not found.".format(results_fn))
            continue
        
        for dependent_var in dependent_vars:
            fig, axs = plt.subplots(ncols=len(parameters), figsize=(20, 10), sharey=True)
            
            fig.suptitle("{} {} ({})".format(dataset, grouping, dependent_var))
                
            group_splits = results.groupby(parameters)
            
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
                ax.set_ylabel(dependent_var)
                
                for other_values, subset in mean_groups:
                    label = ", ".join("{}={}".format(k,v) for k, v in zip(other_parameters, other_values))
                    subset.plot(parameter, dependent_var,
                                ax=ax,
    #                            yerr = std_groups[other_values],
                                xticks=subset[parameter],
                                label=label)
                    
            figure_fname = os.path.join(results_root, "RF_{}_{}_{}.pdf".format(dataset, grouping, dependent_var))
            fig.savefig(figure_fname, bbox_inches="tight")