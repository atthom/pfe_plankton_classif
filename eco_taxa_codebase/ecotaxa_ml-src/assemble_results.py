#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assemble the results from the various experiments and write them into aggregated_results.csv.

TODO: Random Forest (ZP+SCN)

@author: mschroeder
"""

import numpy as np
import os
from util.datasets import DATASETS, GROUPINGS, SPLITS
import pandas as pd

results_root = "/datapc/ob/mschroeder/Ecotaxa/Results"

def scn_mean_val(dataset, grouping):
    last_accs = []
    for split in SPLITS:
        #print(" ", split)
        results_fn = "SCN_{}_{}_{}/wp2.csv".format(dataset, grouping, split)
        #print("  ", results_fn)
        results_fn = os.path.join(results_root, results_fn)
        
        try:
            results = pd.read_csv(results_fn)
        except FileNotFoundError:
            print("  ", "Not found")
            continue 
        
        acc = (100 - results["test_3_mistakes"]) / 100
        
        #max_acc_idx = acc.argmax()
        #print("   Maximum accuracy in epoch {:d}: {:.2%}".format(results["epoch"][max_acc_idx], acc[max_acc_idx]))
        
        last_epoch_idx = results["epoch"].argmax()
        #print("   Accuracy in last epoch {:d}: {:.2%}".format(results["epoch"][last_epoch_idx], acc[last_epoch_idx]))
        
        last_accs.append(acc[last_epoch_idx])
    #print("  SCN (Train): Mean validation accuracy in last epoch {:.2%}".format(np.mean(last_accs)))

    return {"scn_val_acc": np.mean(last_accs),
            "scn_val_epochs": results["epoch"][last_epoch_idx] + 1}

# TODO: Make SCN output _train_confusion.csv
def scn_test_train(dataset, grouping):
    test_confusion_fn = "SCN_{}_{}_TEST/_train_confusion.csv".format(dataset, grouping)
    test_confusion_fn = os.path.join(results_root, test_confusion_fn)
    
    try:
        test_confusion = np.loadtxt(test_confusion_fn)
    except FileNotFoundError:
        print("  ", "Not found:", test_confusion_fn)
        return {}
    
    n_true = np.sum(np.diagonal(test_confusion))
    n_all = np.sum(test_confusion)
    
    acc = n_true / n_all
    
    return {"scn_test_train_acc": acc}

def scn_test_test(dataset, grouping):
    test_confusion_fn = "SCN_{}_{}_TEST/_test_confusion.csv".format(dataset, grouping)
    test_confusion_fn = os.path.join(results_root, test_confusion_fn)
    
    try:
        test_confusion = np.loadtxt(test_confusion_fn)
    except FileNotFoundError:
        print("  ", "Not found:", test_confusion_fn)
        return {}
    
    n_true = np.sum(np.diagonal(test_confusion))
    n_all = np.sum(test_confusion)
    
    acc = n_true / n_all
    
    return {"scn_test_acc": acc}
    
def rf_zp_test(dataset, grouping):
    results_fn = "RF_{}_{}_TEST/random_forest.log".format(dataset, grouping)
    #print("  ", results_fn)
    results_fn = os.path.join(results_root, results_fn)
    
    results = {}
    
    try:
        with open(results_fn, "r") as f:
            parse = False
            for line in f:
                line = line.strip()
                
                if line == "# Results":
                    parse = True
                    continue
                if parse:
                    if ":" not in line:
                        break
                    key, value = line.split(":")
                    key, value = key.strip(), value.strip()
                    
                    if " " in key:
                        break
                    
                    results["rf_zp_test_" + key] = float(value)
    except FileNotFoundError:
        print("  ", "Not found:", results_fn)
        return {} 
    
    return results

def rf_zp_mean_val(dataset, grouping):
    parameters = { "n_estimators": 300, "min_samples_leaf": 2, "max_features": 11}
    output_parameters = ["time_fit_train", "time_predict_train", "score_train", "time_predict_val", "score_val"]
    
    results_fn = "RF_{}_{}/results.csv".format(dataset, grouping)
    results_fn = os.path.join(results_root, results_fn)
    
    try:
        results = pd.read_csv(results_fn)
    except FileNotFoundError:
        print("  ", "Not found:", results_fn)
        return {}
        
    for k, v in parameters.items():
        results = results.where(results[k] == v)
            
    results = results.dropna(0, "all")
    
    if results.empty:
        print("  ", "No results for given parameters:", results_fn)
        return {}
        
    results = results.mean(0)
    
    return {"rf_zp_val_" + k: results[k] for k in output_parameters}

def rf_scn_mean_val(dataset, grouping):
    parameters = { "n_estimators": 300, "min_samples_leaf": 2, "n_components":30}
    output_parameters = ["time_fit_train", "time_predict_train", "score_train", "time_predict_val", "score_val"]
    
    overall_results = []
    
    for split in SPLITS:
        results_fn = "RF+SCN_{}_{}_{}/results.csv".format(dataset, grouping, split)
        results_fn = os.path.join(results_root, results_fn)
        
        try:
            results = pd.read_csv(results_fn)
        except FileNotFoundError:
            print("  ", "Not found:", results_fn)
            continue
        
        for k, v in parameters.items():
            results = results.where(results[k] == v)
            
        results = results.dropna()
        
        if results.empty:
            continue
        
        overall_results.append(results)
        
    if len(overall_results) is 0:
        print("  ", "No results:", results_fn)
        return {}
        
    overall_results = pd.concat(overall_results)
    overall_results = overall_results.mean(0)
    
    return {"rf_scn_val_" + k: overall_results[k] for k in output_parameters}

def rf_scn_test(dataset, grouping):
    results_fn = "RF_SCN_{}_{}_TEST/rf.log".format(dataset, grouping)
    #print("  ", results_fn)
    results_fn = os.path.join(results_root, results_fn)
    
    results = {}
    
    try:
        with open(results_fn, "r") as f:
            parse = False
            for line in f:
                line = line.strip()
                
                if line == "# Results":
                    parse = True
                    continue
                if parse:
                    if ":" not in line:
                        break
                    key, value = line.split(":")
                    key, value = key.strip(), value.strip()
                    
                    if " " in key:
                        break
                    
                    results["rf_scn_test_" + key] = float(value)
    except FileNotFoundError:
        print("  ", "Not found:", results_fn)
        return {} 
    
    return results

def rf_zpscn_mean_val(dataset, grouping):
    parameters = { "n_estimators": 300, "min_samples_leaf": 2}
    output_parameters = ["time_fit_train", "time_predict_train", "score_train", "time_predict_val", "score_val"]
    
    overall_results = []
    
    for split in SPLITS:
        results_fn = "RF_ZP+SCN_{}_{}_{}/results.csv".format(dataset, grouping, split)
        results_fn = os.path.join(results_root, results_fn)
        
        try:
            results = pd.read_csv(results_fn)
        except FileNotFoundError:
            print("  ", "Not found:", results_fn)
            continue
        
        for k, v in parameters.items():
            results = results.where(results[k] == v)
            
        results = results.dropna(0, "all")
        
        if results.empty:
            continue
        
        overall_results.append(results)
        
    if len(overall_results) is 0:
        print("  ", "No results:", results_fn)
        return {}
        
    overall_results = pd.concat(overall_results)
    overall_results = overall_results.mean(0)
    
    return {"rf_zpscn_val_" + k: overall_results[k] for k in output_parameters}

def rf_zpscn_test(dataset, grouping):
    parameters = { "n_estimators": 300, "min_samples_leaf": 2}
    output_parameters = ["time_fit_train", "time_predict_train", "score_train", "time_predict_test", "score_test"]
    
    results_fn = "RF_ZP+SCN_{}_{}_TEST/results.csv".format(dataset, grouping)
    results_fn = os.path.join(results_root, results_fn)
    
    try:
        results = pd.read_csv(results_fn)
    except FileNotFoundError:
        print("  ", "Not found:", results_fn)
        return {}
    
    for k, v in parameters.items():
        results = results.where(results[k] == v)
        
    results = results.dropna(0, "all")
    
    if len(results) == 0:
        print("  ", "No results:", results_fn)
        return {}
    
    return {"rf_zpscn_test_" + k: float(results[k]) for k in output_parameters}
    

results = []
for dataset in DATASETS:
    print(dataset)
    for grouping in GROUPINGS:
        print("", grouping)
        
        result = {"dataset": dataset, "grouping": grouping}
        
        # SCN Validation
        result.update(scn_mean_val(dataset, grouping))
        
        # SCN Testing
        result.update(scn_test_test(dataset, grouping))
        
        # RF (ZooProcess) Validation
        result.update(rf_zp_mean_val(dataset, grouping))
        
        # RF (ZooProcess) Testing
        result.update(rf_zp_test(dataset, grouping))
        
        # RF (SCN) Validation
        result.update(rf_scn_mean_val(dataset, grouping))
        
        # RF (SCN) Test
        result.update(rf_scn_test(dataset, grouping))
        
        # RF (ZP+SCN) Validation
        result.update(rf_zpscn_mean_val(dataset, grouping))
        
        result.update(rf_zpscn_test(dataset, grouping))
        
        results.append(result)
        
results = pd.DataFrame(results)
results.to_csv(os.path.join(results_root, "aggregated_results.csv"))

def latex_report_validation_score(results, path):
    # Format as latex
    format_perc = lambda x: "NaN" if np.isnan(x) else "{:.2%}".format(x)
    
    latex_columns = [("dataset", "Dataset", str),
                     ("grouping", "Grouping", str),
                     ("rf_zp_val_score_val", "Random Forest (ZP)", format_perc),
                     ("rf_scn_val_score_val", "Random Forest (SCN)", format_perc),
                     ("rf_zpscn_val_score_val", "Random Forest (ZP+SCN)", format_perc),
                     ("scn_val_acc", "SparseConvNet", format_perc)]
    
    results = results[list(c[0] for c in latex_columns)]
    results = results.rename(columns = {c[0]: c[1] for c in latex_columns})
    
    results.to_latex(path,
                     index=False,
                     formatters={c[1]: c[2] for c in latex_columns})
    
def latex_report_test_score(results, path):
    # Format as latex
    format_perc = lambda x: "NaN" if np.isnan(x) else "{:.2%}".format(x)
    
    latex_columns = [("dataset", "Dataset", str),
                     ("grouping", "Grouping", str),
                     ("rf_zp_test_score_test", "Random Forest (ZP)", format_perc),
                     ("rf_scn_test_score_test", "Random Forest (SCN)", format_perc),
                     ("rf_zpscn_test_score_test", "Random Forest (ZP+SCN)", format_perc),
                     ("scn_test_acc", "SparseConvNet", format_perc)
                    ]
    
    results = results[list(c[0] for c in latex_columns)]
    results = results.rename(columns = {c[0]: c[1] for c in latex_columns})
    
    results.to_latex(path,
                     index=False,
                     formatters={c[1]: c[2] for c in latex_columns})
        
    
latex_report_validation_score(results, os.path.join(results_root, "validation_scores.tex"))
latex_report_test_score(results, os.path.join(results_root, "test_scores.tex"))
