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

    def load_achitecture(self):
        level = self.get_next_level([None])
        while len(level) != len(self.get_next_level(level)):
            level = self.get_next_level(level)
        print(len(level))
        str_level = []
        for node in level:
            str_level.append(str(node.name))
        print(str_level)

        return load_model("model41.h5"), str_level

    def classify(self, path):
        print("load achitecture...")
        model = self.load_achitecture()

        for ll in os.listdir(path)[0:10]:
            print("img", ll, "...")
            # this is a PIL image
            img = load_img(path + separator + ll, grayscale=True)
            img = img.resize((150, 150), Image.ANTIALIAS)
            img = img_to_array(img)
            img = img.reshape((1,) + img.shape)
            print("classify...")
            answer = self.get_answer(models, labels, img)
            print(answer)

    def get_answer(self, model, labels, img):
        return labels[model.predict_classes(img, batch_size=1, verbose=0)[0]]

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
        model = self.create_top_model()
        self.add_layer_and_train(model, root, nb_batch, nb_epoch, True)

    def add_layer_and_train(self, model, nodes_upper, nb_batch, nb_epoch, first=False):
        next_level = self.get_next_level(nodes_upper)
        print("Add Layer for", len(next_level), "classes...")
        if os.path.exists("model" + str(len(next_level)) + ".h5"):
            print("model exists, loading it...")
            model = load_model("model" + str(len(next_level)) + ".h5")
            self.add_layer_and_train(model, next_level, nb_batch, nb_epoch)
        else:
            str_level = []
            for node in next_level:
                path = separator.join([str(n.name) for n in node.path])
                str_level.append(path)
            model = self.add_layer_model(model, len(str_level), first)
            self.train_manual(model, str_level, nb_batch, nb_epoch)
            save_model(model, "model" + str(len(str_level)) + ".h5")
            if len(next_level) != len(nodes_upper):
                self.add_layer_and_train(model, next_level, nb_batch, nb_epoch)

    def get_next_level(self, list_nodes):
        next_level = []
        for node in list_nodes:
            next_nodes = findall(self.tree, lambda n: n.parent == node)
            if not next_nodes:
                next_level.append(node)
            next_level.extend(next_nodes)
        return next_level

    def add_layer_model(self, model, nb_classes, first):
        if not first:
            model.layers.pop()
            model.layers.pop()
            model.layers.pop()
            model.layers.pop()
            model.layers.pop()
            model.layers.pop()
            model.layers.pop()
            for i in range(len(model.layers) - 1):
                model.layers[i].trainable = False

        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(nb_classes, activation='sigmoid'))
        model.compile(loss=losses.categorical_crossentropy,
                      optimizer=Adam(lr=0.00001), metrics=[metrics.categorical_accuracy])
        return model

    def train_manual(self, model, list_dir, nb_batch, nb_epoch):
        img_loader = ImageLoaderMultiPath(list_dir, grayscale=True)
        nb = img_loader.nb_files // (nb_batch)
        for j in range(nb_epoch):
            print("Epoch", j + 1, "...")
            x, y = img_loader.load(nb_batch)
            model.fit(x, y, batch_size=1,
                      epochs=1, validation_split=0.2)

    def create_top_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=(150, 150, 1),
                         activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), padding='same',
                         activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), padding='same',
                         activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3), padding='same',
                         activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        return model

    def create_top_model_imgnet(self):
        input_tensor = Input(shape=(150, 150, 1))
        base_model = applications.VGG16(
            include_top=False, weights='imagenet', input_tensor=input_tensor)
        base_model.trainable = False

        model = Sequential()
        model.add(base_model)
        model.add(Flatten(input_shape=base_model.output_shape[1:]))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        return model
