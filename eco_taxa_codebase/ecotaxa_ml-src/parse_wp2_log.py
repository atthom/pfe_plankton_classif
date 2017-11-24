#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse the logfile that gets created during SparseConvNet training
and extract training progress and environment variables.

@author: moi
"""

import re
import csv
import os
from argparse import ArgumentParser
import sys

re_epoch = re.compile(r"epoch: (?P<current_epoch>\d+) lr: (?P<current_lr>.+)")
re_progress = re.compile(r"(?P<identifier>.*) (sample|subset)( rep (?P<repetition>\d)/\d)? Mistakes:( )?(?P<mistakes>\d+(.\d+)?)(?:%) NLL( |:)(?P<nll>\d+(.\d+)?)")
re_env_vars = re.compile(r"(?P<DATASET>flowcam|uvp5_?ccelter|zoocam|zooscan)_(?P<GROUPING>group[12])_collection_(?P<SPLIT>[012])")

VARIABLES = {"Start epoch": "START_EPOCH",
             "Stop epoch": "STOP_EPOCH",
             "Initial learning rate": "LEARNING_RATE",
             "Learning rate decay": "DECAY",
             "Training data directory": "TRAINING_DATA_DIR",
             "Validation data directory": "VALIDATION_DATA_DIR",
             "Unlabeled data": "TEST_DATA_DIR",
             "Experiment name": "RESULTS_DIR",
             "Class list": "CLASS_LIST_FN"}

def main():
    # Setup argument parser
    parser = ArgumentParser(
            description="""Parse wp2 log and create structured csv""")
    parser.add_argument("log", help="Path to logfile")

    args = parser.parse_args()
    
    print("Arguments:")
    for arg, val in vars(args).items():
        print(" {}: {}".format(arg, val))
    print()
    
    log_name = os.path.splitext(args.log)[0]
    
    tab_fn = log_name + ".csv"
    options_fn = log_name + ".cfg"
    
    options = {}

    with open(args.log) as f:
        current_epoch = None
        current_row = {}
        
        rows = []
        
        for line in f:
            match_epoch = re_epoch.match(line)
            if match_epoch is not None:
                current_epoch = int(match_epoch.group("current_epoch"))
                
                if len(current_row) > 0:
                    rows.append(current_row)
                
                current_row = {"epoch": current_epoch,
                               "current_lr": match_epoch.group("current_lr")}
                continue
                
            if current_epoch is None:
                # Parse information
                if ":" in line:
                    key, value = line.split(":")
                    options[key.strip()] = value.strip()
            else:
                match_progress = re_progress.match(line)
                
                if match_progress is not None:
                    if match_progress.group("repetition") is not None:
                        prefix = 'test_' + match_progress.group("repetition") + '_'
                    else:
                        prefix = 'train_'
                
                    for key in ["mistakes", "nll"]:
                        current_row[prefix + key] = match_progress.group(key)
                else:
                    #print("Unmatched line:", line)
                    pass
    
    # Append last row
    rows.append(current_row)
    
    keys = set()
    for r in rows:
        keys.update(r.keys())
        
    keys = sorted(keys)
    
    with open(tab_fn, "w") as f:
        writer = csv.DictWriter(f, keys)
        
        writer.writeheader()
        writer.writerows(rows)
        
    # Extract ENV variables
    variables = {VARIABLES[k]: v for k, v in options.items() if k in VARIABLES}
    
    # Parse TRAINING_DATA_DIR
    match_env_vars = re_env_vars.search(variables["TRAINING_DATA_DIR"])
    if match_env_vars is None:
        print("! Could not extract DATASET, GROUPING and SPLIT from TRAINING_DATA_DIR: {}".format(variables["TRAINING_DATA_DIR"]))
    else:
        for k in ["DATASET", "GROUPING", "SPLIT"]:
            variables[k] = match_env_vars.group(k)
            
    if "DATASET" in variables and variables["DATASET"] == "uvp5_ccelter":
        variables["DATASET"] = "uvp5ccelter"
    
    # Write ENV variables
    with open(options_fn, "w") as f:
        for k, v in sorted(variables.items()):
            f.write("{}={}\n".format(k, v))
        
if __name__ == "__main__":
    sys.exit(main())