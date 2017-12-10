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
from ImageLoader import ImageLoaderMultiPath

separator = os.sep


class LayerClassifier:
    def __init__(self, super_path):
        self.super_path = super_path
        self.tree = self.create_tree()

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

    def create_achitecture(self, nb_batch, nb_epoch):
        root = self.get_next_level([None])
        model = self.create_model(len(self.super_path))
        self.add_layer_and_train(model, root, nb_batch, nb_epoch)

    def add_layer_and_train(self, model, nodes_upper, nb_batch, nb_epoch):
        next_level = self.get_next_level(nodes_upper)
        str_level = []
        for node in next_level:
            path = separator.join([str(n.name) for n in node.path])
            str_level.append(path)
        model = self.add_layer_model(model, len(str_level))
        self.train_manual(model, str_level, nb_batch, nb_epoch)
        save_model(model, "model" + str(len(str_level)) + ".h5")
        if next_level:
            self.add_layer_and_train(model, next_level, nb_batch, nb_epoch)

    def get_next_level(self, list_nodes):
        next_level = []
        for node in list_nodes:
            next_nodes = findall(self.tree, lambda n: n.parent == node)
            if not next_nodes:
                next_level.append(node)
            next_level.extend(next_nodes)
        return next_level

    def add_layer_model(self, model, nb_classes):
        model.trainable = False
        new_model = Sequential()
        new_model.add(model)
        new_model.add(Dense(nb_classes, activation='sigmoid'))
        new_model.compile(loss=losses.categorical_crossentropy,
                          optimizer=Adam(lr=0.00001), metrics=[metrics.categorical_accuracy])
        return new_model

    def train_manual(self, model, list_dir, nb_epoch, nb_batch):
        img_loader = ImageLoaderMultiPath(list_dir)
        nb = img_loader.nb_files // (nb_batch)
        for j in range(nb_epoch):
            x, y = img_loader.load(nb_batch)
            model.fit(x, y, batch_size=nb_batch,
                      epochs=1, validation_split=0.2)

    def create_model(self, nb_classes):
        input_tensor = Input(shape=(150, 150, 3))
        base_model = applications.VGG16(
            include_top=False, weights='imagenet', input_tensor=input_tensor)
        model = Sequential()
        model.add(base_model)
        model.add(Flatten(input_shape=base_model.output_shape[1:]))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))

        return model
