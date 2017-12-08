import os
super_path = "E:\\Polytech_Projects\\pfe_plankton_classif\\LOOV\\super_classif"
import json
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import PIL
import numpy as np


class ImageLoader:
    def __init__(self, super_path):
        self.super_path = super_path
        self.database = self.create_dict()
        self.id = 0

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

    def load(self, nb_imgs, sparse=1):
        nb_load = int(nb_imgs * sparse)
        keys = list(self.database.keys())[self.id:self.id + nb_load]
        self.id += nb_imgs
        x_pred = []
        y_pred = []
        for path in keys:
            img = np.asarray(PIL.Image.open(path))
            img = img.resize((150, 150), Image.ANTIALIAS)
            img = img.reshape((1,) + img.shape)
            x_pred.append(img)
            y_pred.append(self.database[path])
        return x_pred, y_pred


loader = ImageLoader(super_path)
loader.load(100)
