import os
import json
import numpy as np
from PIL import Image
from anytree import Node, RenderTree
from anytree.search import findall
from keras.engine import Model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential, save_model, load_model
from keras.layers import Dropout, Flatten, Dense, MaxPooling2D, Conv2D
from keras import applications, Input, losses, metrics
from keras.optimizers import Adam
from ImageLoader import ImageLoader
from keras.layers import LeakyReLU

separator = os.sep


class TreeClassifier:
    def __init__(self, super_path, load_model=False):
        self.super_path = super_path
        self.tree = self.create_tree()
        self.directories = self.get_directories()

        if load_model:
            print("Loading models...")
            self.models, self.labels = self.load_architecture()

    def create_achitecture(self, datagen, nb_epoch, batch_size):
        for super_dir in self.directories:
            self.train_model(datagen, super_dir, nb_epoch, batch_size)

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

    def get_directories(self):
        nodes = []
        for pre, fill, node in RenderTree(self.tree):
            path = [str(_.name) for _ in node.path]
            if len(path) > 1:
                nodes.append(path[-2])

        nodes = list(set(nodes))
        path_nodes = []

        for node in nodes:
            nn = findall(self.tree, lambda n: node == n.name)[0]
            path = separator.join([str(node.name) for node in nn.path])
            path_nodes.append(path)

        path_nodes = sorted(
            path_nodes, key=lambda path: len(path.split(separator)))
        return path_nodes

    def print_tree(self):
        for pre, fill, node in RenderTree(self.tree):
            print("%s%s" % (pre, node.name))

    def train_model(self, datagen, data_dir, nb_epoch, batch_size):
        model = self.create_model(len(os.listdir(data_dir)), data_dir)
        nb_img = sum([len(files) for r, d, files in os.walk(data_dir)])

        train_generator = datagen.flow_from_directory(
            data_dir, batch_size=batch_size, target_size=(150, 150), color_mode='grayscale')

        nb_step = 300
        # nb_step = nb_img // batch_size
        nb_validation = nb_step // 10

        model.fit_generator(train_generator,
                            steps_per_epoch=nb_step,
                            validation_data=train_generator,
                            validation_steps=nb_validation,
                            epochs=nb_epoch,
                            workers=4)
        data_dir = data_dir.split(separator)[-1]
        save_model(model, "./model_full_datagen/model_" + data_dir + ".h5")

        # Writing JSON data
        with open("./model_full_datagen/labels_" + data_dir + ".json", "w") as f:
            json.dump(train_generator.class_indices, f)

    def create_manual_all(self, nb_batch, nb_epoch):
        for super_dir in self.directories:
            print(super_dir, "...")
            img_loader = ImageLoader(super_dir, already_formated=True)

            dd = super_dir.split(separator)[-1]
            if os.path.exists("./model_full_datagen/model_" + dd + ".h5"):
                continue

            model = self.create_model(len(os.listdir(super_dir)), super_dir)
            x, y = img_loader.load_all()

            for i in range(nb_epoch // 2):
                model.fit(x, y, batch_size=nb_batch,
                          epochs=2, validation_split=0.2)
                dd = super_dir.split(separator)[-1]
                save_model(model, "./model_full_datagen/model_" + dd + ".h5")
                with open("./model_full_datagen/labels_" + dd + ".json", "w") as f:
                    json.dump(os.listdir(super_dir), f)

    def create_manual(self, nb_batch, nb_epoch):
        for super_dir in self.directories:
            print(super_dir, "...")
            img_loader = ImageLoader(super_dir, already_formated=True)
            dd = super_dir.split(separator)[-1]
            if os.path.exists("./model_full_datagen/model_" + dd + ".h5"):
                continue
            model = self.create_model(len(os.listdir(super_dir)), super_dir)
            nb = img_loader.nb_files // (nb_batch)

            for i in range(1):
                x, y = img_loader.load(nb_batch, super_dir)
                model.fit(x, y, batch_size=1, epochs=3, validation_split=0.2)

                dd = super_dir.split(separator)[-1]
                save_model(model, "./model_full_datagen/model_" + dd + ".h5")
                with open("./model_full_datagen/labels_" + dd + ".json", "w") as f:
                    json.dump(os.listdir(super_dir), f)

    def create_model(self, nb_classes, super_dir):
        dd = super_dir.split(separator)[-1]
        # if os.path.exists("./model_cluster/model_" + dd + ".h5"):
        #    return load_model("./model_cluster/model_" + dd + ".h5")

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=(150, 150, 1)))
        model.add(LeakyReLU(alpha=(1 / 3)))
        model.add(Conv2D(16, (3, 3)))
        model.add(LeakyReLU(alpha=(1 / 3)))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(LeakyReLU(alpha=(1 / 3)))
        model.add(Conv2D(32, (3, 3)))
        model.add(LeakyReLU(alpha=(1 / 3)))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(LeakyReLU(alpha=(1 / 3)))
        model.add(Conv2D(128, (3, 3)))
        model.add(LeakyReLU(alpha=(1 / 3)))
        model.add(Conv2D(64, (3, 3)))
        model.add(LeakyReLU(alpha=(1 / 3)))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(LeakyReLU(alpha=(1 / 3)))
        model.add(Conv2D(256, (3, 3)))
        model.add(LeakyReLU(alpha=(1 / 3)))
        model.add(Conv2D(128, (3, 3)))
        model.add(LeakyReLU(alpha=(1 / 3)))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=(1 / 3)))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))

        model.compile(loss=losses.categorical_crossentropy,
                      optimizer=Adam(lr=0.000001),
                      metrics=[metrics.categorical_accuracy])
        return model

    def load_architecture(self):
        models = dict()
        labels = dict()
        for data_dir in self.directories:
            dd = data_dir.split(separator)[-1]
            models[dd] = load_model("./model_full_datagen/model_" + dd + ".h5")
            with open("./model_full_datagen/labels_" + dd + ".json", "r") as f:
                labels[dd] = json.load(f)
        return models, labels

    def classify(self, img):
        first = self.directories[0].split(separator)[-1]

        answers = [self.get_answer(
            self.models[first], self.labels[first], img)]

        while answers[-1] in self.models.keys():
            c_model = self.models[answers[-1]]
            c_label = self.labels[answers[-1]]
            answers.append(self.get_answer(c_model, c_label, img))
        return answers

    def get_answer(self, model, labels, img):
        #print(labels[model.predict_classes(img, batch_size=1, verbose=0)[0]])
        return labels[model.predict_classes(img, batch_size=1, verbose=0)[0]]

    def get_next_level(self, list_nodes):
        next_level = []
        for node in list_nodes:
            next_nodes = findall(self.tree, lambda n: n.parent == node)
            if not next_nodes:
                next_level.append(node)
            next_level.extend(next_nodes)
        return next_level

    def get_last_level(self):
        nodes = self.get_next_level([None])
        while len(self.get_next_level(nodes)) != len(nodes):
            nodes = self.get_next_level(nodes)
        names = []
        for n in nodes:
            names.append(n.name)
        return names
