from keras.preprocessing.image import ImageDataGenerator
from TreeClassifier import TreeClassifier
from keras.engine import Model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, save_model, load_model
from keras.layers import Dropout, Flatten, Dense, MaxPooling2D, Conv2D
from keras import applications, Input, losses, metrics
from keras.optimizers import Adam
import numpy as np
import os
import json
from PIL import Image

separator = os.sep

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    # zca_whitening=True,
    # zca_epsilon=1e-6,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    # channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=None  # On pourrai peut être resize ici
    # data_format=K.image_data_format()
)

super_path = "E:\\Polytech_Projects\\pfe_plankton_classif\\LOOV\\super_classif"
super_path = "E:\\Polytech_Projects\\pfe_plankton_classif\\Dataset\\DATASET\\level0_new_hierarchique"

classifier = TreeClassifier(super_path)

#classifier.create_achitecture(datagen, nb_epoch=20, batch_size=1)

classifier.create_manual_all(nb_batch=128, nb_epoch=20)
