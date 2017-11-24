#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use Popen to call a SparseConvNet binary

Not fully functional, only a template for the implementation in Ecotaxa.

@author: mschroeder
"""

from subprocess import Popen, TimeoutExpired, DEVNULL, PIPE
from tempfile import mkstemp, mkdtemp
import os
import glob
from itertools import islice
import shutil
import csv
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import numpy as np

SCN_BINARY = "/home/mschroeder/Ecotaxa/SparseConvNet/ecotaxa"
LD_LIBRARY_PATH = os.environ["LD_LIBRARY_PATH"]

def load_features(features_fn):
    """
    Don't use np.loadtxt or np.genfromtxt, as these won't preallocate an array
    and instead assemble a list of values in an inefficient way.
    """
    
    delimiter=","
    
    with open(features_fn) as f:
        lines = list(f)
        
    n_cols = len(lines[0].rstrip("\n").split(delimiter))
    n_rows = len(lines)
    
    # Allocate an array of the right size
    dtype = np.dtype([("objid", "<U32"), ("y", np.int), ("X", np.float, n_cols - 2)])
    data = np.empty(n_rows, dtype)
    
    for i, line in enumerate(lines):
        objid, y, *X = line.rstrip("\n").split(delimiter)
        data["objid"][i] = objid
        data["y"][i] = int(y)
        data["X"][i] = np.array(X, np.float)
        
    return data

def compute_features(model_dir, unlabeled_data_fn):
    env = {
       # MODEL_DIR contains the weights (_epoch-X.cnn), and the class file (classes.txt) 
       "MODEL_DIR": "/home/mschroeder/Ecotaxa/Results/SCN_flowcam_group1",
       # OUTPUT_DIR is an existing temporary directory, where the results are put
       # After computation, it will contain one or more of {training,validation,testing,unlabeled}_{predictions,confusion,features}.csv
       "OUTPUT_DIR": output_dir,
       # "CUDA_VISIBLE_DEVICES": "GPU-89525854",
       "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
       "DUMP_FEATURES": "1",
       "UNLABELED_DATA_FN": unlabeled_data_fn
    }
    
    for k, v in env.items():
        print("{}: {}".format(k,v))
        

    with Popen(SCN_BINARY, env=env) as p:
        returncode = p.wait()
    
    if returncode != 0:
        print("There were errors.")
        return
    
def train_model(model_dir, training_data_fn, testing_data_fn=None, pca_components=50, env={}):
    """
    model_dir has to exist.
    """
    output_dir = mkdtemp(prefix="scn_output_")
    
    # Generate classes.txt from train_data_fn
    classes = set()
    with open(training_data_fn) as f:
        reader = csv.reader(f, delimiter=",")
        
        for objid, image_path, label in reader:
            classes.add(label)
            
    classes = sorted(classes)
    
    classes_fn = os.path.join(model_dir, "classes.txt")
    with open(classes_fn, "w") as f:
        f.writelines(cls + "\n" for cls in classes)
    
    env = dict({
       # MODEL_DIR contains the weights (_epoch-X.cnn), and the class file (classes.txt) and the PCA
       "MODEL_DIR": model_dir,
       # OUTPUT_DIR is an existing temporary directory, where the results are put
       # After running SCN, it will contain one or more of {training,validation,testing,unlabeled}_{predictions,confusion,features}.csv
       "OUTPUT_DIR": output_dir,
       "DUMP_FEATURES": "1",
       "TRAINING_DATA_FN": training_data_fn,
       "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
       "EPOCH": 0,
       "STOP_EPOCH": 100,
    }, **env)
        
    if testing_data_fn is not None:
        env["TESTING_DATA_FN"] = testing_data_fn
        
    env = {k: str(v) for k, v in env.items()}
    
    for k, v in env.items():
        print("{}: {}".format(k,v))

    print("Training SparseConvNet...")
    with Popen(SCN_BINARY, env=env) as p:
        returncode = p.wait()
    
    if returncode != 0:
        print("There were errors.")
        return
    
    # model_dir should now contain the weight snapshots
    # It is save to remove all but the last (_epoch-99.cnn)
    
    # output_dir should now contain the features for training and testing data
    
    # Load training features
    print("Loading features...")
    training_features_fn = os.path.join(output_dir, "training_features.csv")
    training_features = load_features(training_features_fn)

    # Fit and save PCA
    print("Fitting PCA...")
    pca = PCA(n_components = pca_components)
    pca.fit(training_features["X"])
    pca_fn = os.path.join(model_dir, "pca.jbl")
    joblib.dump(pca, pca_fn)
    
    # Model dir should now contain the CNN weights (_epoch-X.cnn), and the class file (classes.txt), and the PCA model (pca.jbl)
    
    # output_dir can now be removed:
    # shutil.rmtree(output_dir)

def do_train(args):
    train_model(args.model_dir, args.training_data_fn, pca_components=args.pca_components)
    
def do_compute_features(args):
    ...    
        
def main():
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="""
    Create a image collection based on a collection index.
    """)
    subparsers = parser.add_subparsers()
    
    parser_train = subparsers.add_parser('train', help="Train a model")
    parser_train.add_argument("model_dir")
    parser_train.add_argument("training_data_fn")
    parser_train.add_argument("--pca-components", type=int, default=50)
    parser_train.set_defaults(func=do_train)

    args = parser.parse_args()
    args.func(args)
    
if __name__ == "__main__":
    main()
    
#_, unlabeled_data_fn = mkstemp(prefix="scn_unlabeled_", suffix=".csv")
#output_dir = mkdtemp(prefix="scn_output_")
#
## Load 100 images. In ecotaxa this should come from the database
#image_path = "/home/mschroeder/Ecotaxa/Datasets/flowcam/*/*.jpg"
#objects = [(str(objid), path) for objid, path in enumerate(islice(glob.iglob(image_path), 100))]
#
## Fill input_fn with data
#with open(unlabeled_data_fn, "w") as f:
#    for obj in objects:
#        f.write(",".join(obj) + "\n")
#    
#_, unlabeled_data_fn = mkstemp(prefix="scn_unlabeled_", suffix=".csv")
#output_dir = mkdtemp(prefix="scn_output_")
#
## Load 100 images. In ecotaxa this should come from the database
#image_path = "/datapc/ob-ssd/mschroeder/Ecotaxa/Datasets/flowcam/*/*.jpg"
#objects = [(str(objid), path) for objid, path in enumerate(islice(glob.iglob(image_path), 100))]
#
## Fill input_fn with data
#with open(unlabeled_data_fn, "w") as f:
#    for obj in objects:
#        f.write(",".join(obj) + "\n")
#
#env = {
#       # MODEL_DIR contains the weights (_epoch-X.cnn), and the class file (classes.txt) 
#       "MODEL_DIR": "/data1/mschroeder/Ecotaxa/Results/SCN_flowcam_group1",
#       # OUTPUT_DIR is an existing temporary directory, where the results are put
#       # After running SCN, it will contain one or more of {training,validation,testing,unlabeled}_{predictions,confusion,features}.csv
#       "OUTPUT_DIR": output_dir,
#       "CUDA_VISIBLE_DEVICES": "GPU-89525854",
#       "DUMP_FEATURES": "1",
#       "UNLABELED_DATA_FN": unlabeled_data_fn
#}
#
#for k, v in env.items():
#    print("{}: {}".format(k,v))
#
#returncode, outs, errs = call_scn(env)
#
#print("Return code: {}".format(returncode))
#
#if returncode != 0:
#    print("There were errors.")
#        
#print("Output:")
#print(outs)
#
#print()
#print("Errors:")
#print(errs)
#
#if returncode == 0:
#    # Read the output
#    with open(os.path.join(output_dir, "unlabeled_features.csv"), "r") as f:
#        for line in f:
#            line = line.strip()
#            objid, label, *features = line.split(",")
#            
#            #print(objid, features)
#        
## Remove output directory
## shutil.rmtree(output_dir)
#        
## Remove input file
## os.unlink(unlabeled_data_fn)

