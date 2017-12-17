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

separator = os.sep


class TreeClassifier:
    def __init__(self, super_path):
        self.super_path = super_path
        self.tree = self.create_tree()
        self.directories = self.get_directories()

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
        model = self.create_model(len(os.listdir(data_dir)))
        nb_img = sum([len(files) for r, d, files in os.walk(data_dir)])

        train_generator = datagen.flow_from_directory(
            data_dir, batch_size=batch_size, target_size=(150, 150), color_mode='grayscale')

        nb_step = 300
        nb_step = nb_img // batch_size
        nb_validation = nb_step // 10

        model.fit_generator(train_generator,
                            steps_per_epoch=nb_step,
                            validation_data=train_generator,
                            validation_steps=nb_validation,
                            epochs=nb_epoch,
                            workers=4)
        data_dir = data_dir.split(separator)[-1]
        save_model(model, "./model_cluster/model_" + data_dir + ".h5")

        # Writing JSON data
        with open("./model_cluster/labels_" + data_dir + ".json", "w") as f:
            json.dump(train_generator.class_indices, f)

    def create_manual(self, nb_batch, nb_epoch):
        for super_dir in self.directories:
            model = self.create_model(len(os.listdir(super_dir)))
            self.train_manual(model, super_dir, nb_batch, nb_epoch)
            dd = super_dir.split(separator)[-1]
            save_model(model, "./model_cluster/model_" + dd + ".h5")
            with open("./model_cluster/labels_" + dd + ".json", "w") as f:
                json.dump(os.listdir(super_dir), f)

    def train_manual(self, model, super_dir, nb_batch, nb_epoch):
        img_loader = ImageLoader(super_dir)
        nb = img_loader.nb_files // (nb_batch)

        for i in range(10):
            x, y = img_loader.load(nb_batch, super_dir)
            model.fit(x, y, batch_size=1, epochs=3, validation_split=0.2)

    def create_model(self, nb_classes):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=(150, 150, 1),
                         activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(nb_classes, activation='softmax'))

        adam = Adam(lr=0.000001)
        model.compile(loss=losses.categorical_crossentropy,
                      optimizer=adam, metrics=[metrics.categorical_accuracy])

        return model

    def load_architecture(self):
        models = dict()
        labels = dict()
        for data_dir in self.directories:
            dd = data_dir.split(separator)[-1]
            models[dd] = load_model("./model_cluster/model_" + dd + ".h5")
            with open("./model_cluster/labels_" + dd + ".json", "r") as f:
                label = dict((v, k) for k, v in json.load(f).items())
                labels[dd] = label
        return models, labels

    def classify(self, path):
        print("load achitecture...")
        models, labels = self.load_architecture()
        print(models.keys(), labels.keys(), path)

        for ll in os.listdir(path)[0:10]:
            print("img", ll, "...")
            # this is a PIL image
            img = load_img(path + separator + ll, grayscale=True)
            img = img.resize((150, 150), Image.ANTIALIAS)
            img = img_to_array(img)
            img = img.reshape((1,) + img.shape)
            print("classify...")
            answer = self.classify_worker(models, labels, img)
            print(answer)

    def get_answer(self, model, labels, img):
        return labels[model.predict_classes(img, batch_size=1, verbose=0)[0]]

    def classify_worker(self, models, labels, img):
        first = self.directories[0].split(separator)[-1]
        answers = [self.get_answer(models[first], labels[first], img)]
        while answers[-1] in models.keys():
            current_model = models[answers[-1]]
            current_label = labels[answers[-1]]
            answers.append(self.get_answer(current_model, current_label, img))
        return answers
