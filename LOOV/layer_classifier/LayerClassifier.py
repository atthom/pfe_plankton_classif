import os
import json
from PIL import Image
from anytree import Node, RenderTree
from anytree.search import findall
from keras.engine import Model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, save_model, load_model
from keras.layers import Dropout, Flatten, Dense, MaxPooling2D, Conv2D
from keras import applications, Input, losses, metrics
from keras.optimizers import Adam
import numpy as np

separator = os.sep


class LayerClassifier:
    def __init__(self, super_path):
        self.super_path = super_path

    def create_tree(self):
        tree = dict()
        super_dir = self.super_path.split(separator)[-1]
        tree[super_dir] = Node(self.super_path)

        for root, dirs, files in os.walk(self.super_path):
            if root == self.super_path:
                continue
            path = root.split(separator)
            tree[path[-1]] = Node(path[-1], parent=tree[path[-2]])
        return tree[super_dir]

    def get_nb_classes(self):
        tree = self.create_tree()

        nb = findall(tree, lambda n: n.parent)

    def create_achitecture(self, datagen, nb_epoch):
        sub_dirs = os.listdir(self.super_path)
        model = self.create_model(len(sub_dirs)))

        train_generator = datagen.flow_from_directory(
            self.super_path, batch_size=8, target_size=(150, 150), color_mode='grayscale')

        self.train_model(train_generator, model, nb_epoch)

        nb = sum([len(sub) for sub in sub_dirs])
    
    def add_layer(model, nb_classes):
        model.trainable = False

        new_model = Sequential()
        new_model.add(model)
        new_model.add(Dense(nb_classes, activation='sigmoid'))
        new_model.compile(loss=losses.categorical_crossentropy,
                      optimizer=Adam(lr=0.00001), metrics=[metrics.categorical_accuracy])
        return new_model




    def train_model(self, train_generator, model, nb_epoch):

        model.fit_generator(train_generator,
                            # steps_per_epoch=nb_img // batch_size,
                            steps_per_epoch=300,
                            validation_data=train_generator,
                            # validation_steps=nb_img // (2 * batch_size),
                            validation_steps=30,
                            epochs=nb_epoch, workers=4)
        data_dir = self.super_path.split(separator)[-1]
        save_model(model, "models_quick/model_" + data_dir + ".h5")

    def create_model(self, nb_classes):
        model = Sequential()

        model.add(Flatten(input_shape=(150, 150, 1)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(nb_classes, activation='sigmoid'))

        adam = Adam(lr=0.00001)
        model.compile(loss=losses.categorical_crossentropy,
                      optimizer=adam, metrics=[metrics.categorical_accuracy])
        return model
