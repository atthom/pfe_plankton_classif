#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a Keras model.

@author: mschroeder
"""

import os
import socket
import time
from functools import partial

import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from models.vgg16 import VGG16
from util.data import LabelEncoder, CachingImageLoader
from util.sampling import BatchGenerator
from util.sampling import ClassAwareSampling
from util.sampling import RandomSampling
from util.transformation import transform_sample_cv

epochs = 100
batch_size = 48
image_size = 300
n_channels = 1

TRAINING_DATA_INDEX = os.environ["TRAINING_DATA_INDEX"]
VALIDATION_DATA_INDEX = os.environ["VALIDATION_DATA_INDEX"]
CLASS_LIST_FN = os.environ["CLASS_LIST_FN"]
TEST_DATA_DIR = os.environ["TEST_DATA_DIR"]
RESULTS_DIR = os.environ["RESULTS_DIR"]

image_root_dir = os.path.dirname(TRAINING_DATA_INDEX)

print("Arguments:")
for arg in "TRAINING_DATA_INDEX,VALIDATION_DATA_INDEX,CLASS_LIST_FN,TEST_DATA_DIR,RESULTS_DIR".split(","):
    print(" {}: {}".format(arg, globals().get(arg)))
print()

def preprocess_batch(X, y, random_transform=True):
    """
    Parameters:
        X: list of fnames
        y: list of numerical labels
    """

    X_arr = np.empty((len(X), image_size, image_size, n_channels), dtype=K.floatx())
    for i, x in enumerate(X):
        # Read image with filename x
        x = image_loader[x].astype(K.floatx())
        
        assert x.shape[2] == n_channels
        
        x /= 255.0

        X_arr[i] = transform_sample_cv(x, image_size, 1.5)

    # Convert to categorial labels
    y = to_categorical(y, num_classes=num_classes)

    return (X_arr, y)


# Configure session
if socket.gethostname() == "ems":
    gpu_options = {"per_process_gpu_memory_fraction": 0.90,
                   "allow_growth": True}
else:
    gpu_options = {}
gpu_options = tf.GPUOptions(**gpu_options)
tf.logging.set_verbosity(tf.logging.DEBUG)
config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=config)
K.set_session(sess)

# Load classes
classes = np.loadtxt(CLASS_LIST_FN, dtype="U32", delimiter=",", usecols=0)
num_classes = len(classes)
label_encoder = LabelEncoder(classes)

print("%d classes: %s" % (num_classes, ", ".join(classes)))

# Load data
print("Loading dataset...")
time_start = time.time()
X_train, y_train = np.loadtxt(TRAINING_DATA_INDEX,
                              dtype="object,U32", delimiter=",", unpack=True)
y_train = label_encoder.transform(y_train)

X_val, y_val = np.loadtxt(VALIDATION_DATA_INDEX,
                          dtype="object,U32", delimiter=",", unpack=True)
y_val = label_encoder.transform(y_val)
print("Done. (%.2f seconds)" % (time.time() - time_start))
print("{:d} training samples, {:d} validation samples.".format(len(X_train), len(X_val)))

# CachingImageLoader will store a file in memory once loaded
image_loader = CachingImageLoader(root=image_root_dir, n_channels=n_channels)

def preload_data():
    print("Preloading images...")
    time_start = time.time()
    image_loader.preload(X_train)
    image_loader.preload(X_val)
    print(
        "Finished preloading images. (%.2f seconds)" %
        (time.time() - time_start))


#Thread(target=preload_data).start()
preload_data()

# Set up generation of training and validation data

# Sample randomly
#train_idx_gen = RandomSampling(np.arange(len(X_train)))

# Samples are drawn from all classes equally
train_idx_gen = ClassAwareSampling(y_train)

inactive_classes = train_idx_gen.get_inactive_classes(num_classes)
print("Classes ignored during training:",
          ", ".join(label_encoder.inverse_transform(inactive_classes)))


train_batch_gen = BatchGenerator(train_idx_gen, batch_size,
                                 X_train, y_train,
                                 preprocessing_function=preprocess_batch)

# Samples are drawn randomly
val_idx_gen = RandomSampling(np.arange(len(X_val)))
val_batch_gen = BatchGenerator(val_idx_gen, batch_size,
                    X_val, y_val,
                    preprocessing_function=partial(
                            preprocess_batch, random_transform=False))

def dump_batches(batch_gen):
    for batch in batch_gen:
        print(batch)
        yield batch
        
#train_batch_gen = dump_batches(train_batch_gen)


#num_train_batches = len(X_train) // batch_size
#num_val_batches = len(X_val) // batch_size

# Each training epoch should contain num_classes * 1000 samples
num_train_batches = (num_classes * 1000) // batch_size

# Each validation epoch should contain num_classes * 100 samples
num_val_batches = (num_classes * 100) // batch_size

# Initialize model
model = VGG16(num_classes, activation="elu")
#opt = RMSprop(lr=0.001, decay=1e-6)
opt = RMSprop(lr=0.001)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# Callbacks
callbacks = []

callbacks.append(
    ModelCheckpoint(
        monitor='val_loss',
        filepath=os.path.join(
            RESULTS_DIR,
            "weights_best.hdf5"),
        verbose=1,
        save_best_only=True))
        
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5))

callbacks.append(
    CSVLogger(
        os.path.join(
            RESULTS_DIR,
            "log.csv"),
        separator="\t"))
        
print("Training...")
time_start = time.clock()
try:
    model.fit_generator(train_batch_gen, steps_per_epoch=num_train_batches,
                        epochs=epochs,
                        validation_data=val_batch_gen, validation_steps=num_val_batches,
                        callbacks=callbacks)
except KeyboardInterrupt:
    pass
print("Training took %.2f seconds." % (time.clock() - time_start))

weights_fn = os.path.join(RESULTS_DIR, "weights_latest.hdf5")
print("Saving weights to %s..." % weights_fn)
model.save_weights(weights_fn)

print("Done.")
