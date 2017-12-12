import os
super_path = "E:\\Polytech_Projects\\pfe_plankton_classif\\LOOV\\super_classif"
import json
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import PIL
import numpy as np
import random


class ImageLoader:
    def __init__(self, super_path):
        self.super_path = super_path
        self.database = self.create_dict()
        self.nb_files = len(self.database)
        self.keys_used = []

    def create_dict(self):
        if os.path.isfile("database.json"):
            with open("database.json", "r") as f:
                return json.load(f)

        database = dict()
        for d, sub_dir, files in os.walk(self.super_path):
            for file in files:
                label = d.split(os.sep)[-1]
                database[d + os.sep + file] = label
        with open("database.json", "w") as f:
            json.dump(database, f)
        return database

    def load(self, nb_imgs, data_dir):
        sub_dir = os.listdir(data_dir)
        base_anwser = [0] * len(sub_dir)
        keys = list(self.database.keys())
        n_keys = []
        for k in keys:
            if k not in self.keys_used:
                mm = data_dir.split(os.sep)[-1]
                dd = k.split(os.sep)[-3]
                if mm == dd:
                    n_keys.append(k)
        nb_keys = len(n_keys)
        id_picked = random.sample(
            range(nb_keys), min(nb_imgs, nb_keys))
        d_keys = []
        [d_keys.append(keys[id]) for id in id_picked]
        self.keys_used.extend(d_keys)

        x = []
        y = []
        for path in d_keys:
            base_anwser = np.asarray([0] * len(sub_dir))
            i = sub_dir.index(self.database[path])
            base_anwser[i] = 1.
            img = load_img(path, grayscale=True)
            img = img.resize((150, 150), PIL.Image.ANTIALIAS)
            img = img_to_array(img)
            x.append(img)
            y.append(base_anwser)

        return np.array(x), np.array(y)


class ImageLoaderMultiPath:
    def __init__(self, multi_super_path, grayscale):
        self.multi_super_path = multi_super_path
        self.grayscale = grayscale

        self.database = dict()
        for super_path in self.multi_super_path:
            self.database = {**self.database, **self.create_dict(super_path)}

        self.nb_files = len(self.database)
        self.keys_used = []

    def create_dict(self, path):
        database = dict()
        for d, sub_dir, files in os.walk(path):
            for file in files:
                label = d.split(os.sep)[-1]
                database[d + os.sep + file] = label
        return database

    def load(self, nb_imgs):
        keys = list(self.database.keys())
        nb_keys = len(keys)
        id_picked = random.sample(
            range(nb_keys), min(nb_imgs, nb_keys))
        keys_to_use = []
        [keys_to_use.append(keys[id]) for id in id_picked]
        self.keys_used.extend(keys_to_use)

        x = []
        y = []
        for path in keys_to_use:
            base_anwser = np.asarray([0] * len(self.multi_super_path))
            i = 0
            for k in range(len(self.multi_super_path)):
                if self.multi_super_path[k] in path:
                    base_anwser[k] = 1.

            img = load_img(path, grayscale=self.grayscale)
            img = img.resize((150, 150), PIL.Image.ANTIALIAS)
            img = img_to_array(img)
            x.append(img)
            y.append(base_anwser)

        return np.array(x), np.array(y)
