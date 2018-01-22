from keras.models import save_model, load_model
from anytree.search import findall
from anytree import Node, RenderTree
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import metrics, losses
import csv
import json
import PIL
import os
import random
import numpy as np
import tensorflow as tf
from TreeClassifier import TreeClassifier
from scipy.misc import imread, imsave


class Benchmark:
    def __init__(self, path):
        self.path_img = path + "imgs"
        if os.path.exists("database_validation.json"):
            self.database = self.load_database()
        else:
            self.database = self.construct_database(path)

    def construct_database(self, path):
        path_csv = path + "meta.csv"
        all_answer = dict()
        print("Construct dataset...")

        with open(path_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(reader, None)
            i = 0
            for row in reader:
                i += 1
                all_answer[row[0] + ".jpg"] = row[-1]
                if i % 1000 == 0:
                    print("row", i, "...")
        with open("database_validation.json", "w") as f:
            json.dump(all_answer, f)
        return all_answer

    def load_database(self):
        with open("database_validation.json", "r") as f:
            return json.load(f)

    def benchmark_TreeClassifier(self, tree_classifier):
        xx, yy = self.load_all(
            tree_classifier.directories[0], tree_classifier.get_last_level(), number=False)
        nb = len(xx)
        good_guess = 0
        for x, y in zip(xx, yy):
            x = np.reshape(x, (1, 150, 150, 1))
            y_pred = tree_classifier.classify(x)
            print(y_pred[-1], y)
            if y_pred == y:
                good_guess += 1

        print("nb:", nb, "good gess:", good_guess, "accuracy:", good_guess / nb)

    def benchmark_TreeClassifierEachModel(self, tree_classifier):
        benchmark = dict()

        for path in tree_classifier.directories:
            first = path.split(os.sep)[-1]
            print("Benchmark", first)
            model = tree_classifier.models[first]
            labels = tree_classifier.labels[first]
            print("Loading images...")
            x, y = self.load_all(path, labels)
            print("predict on", len(x), "elements...")
            metrics = model.evaluate(x, y, batch_size=64)
            print("loss: %.2f accuracy: %.2f%%" %
                  (metrics[0], metrics[1] * 100))
            benchmark[first] = metrics

        f = tree_classifier.directories[0].split(os.sep)[-1]
        with open("./benchmark" + f + ".json", "w") as f:
            json.dump(benchmark, f)

    def heatmap(self, matrix):
        import matplotlib.pyplot as plt
        plt.imshow(matrix, cmap='hot', interpolation='nearest')
        plt.show()

    def load_all(self, directory, labels, number=True):
        keys = list(self.database.keys())
        random.shuffle(keys)
        x = []
        y = []
        tree = dict()
        for r, d, f in os.walk(directory):
            tree[r] = d

        for file in keys:
            answer = self.database[file]
            if number:
                base_anwser = np.asarray([0] * len(labels))
                if answer not in labels:
                    answer = self.find_answer(answer, labels)
                index = labels.index(answer)
                base_anwser[index] = 1
            else:
                if answer in labels:
                    base_anwser = answer
                else:
                    continue

            img = self.load_image(self.path_img + os.sep + file)
            x.append(img)
            y.append(base_anwser)

            if len(x) % 1000 == 0:
                print(len(x), "images loaded...")

            if len(x) == 102:
                break

        return np.array(x), np.array(y)

    def find_answer(self, answer, labels):
        for r, d in self.tree.items():
            if answer in d:
                r = set(r.split(os.sep))
                return list(r.intersection(set(labels)))[0]

    def load_image(self, path_file, height=150, width=150):
        img = PIL.Image.open(path_file)
        np_img = np.array(img.copy())
        h, w = np_img.shape

        half_w = w // 2
        half_h = h // 2
        half_height = height // 2
        half_width = width // 2

        if w > width:
            if h > height:
                if h / height > w / width:
                    w_compensated = h * width // height
                    final_img = np.ones((h, w_compensated),
                                        dtype="uint8") * 255
                    final_img[0:h, int(w_compensated / 2 - w / 2)                              :int(w_compensated / 2 + w / 2)] = np_img
                else:
                    h_compensated = w * height // width
                    final_img = np.ones((h_compensated, w),
                                        dtype="uint8") * 255
                    final_img[int(h_compensated / 2 - h / 2)
                                  :int(h_compensated / 2 + h / 2), 0:w] = np_img
            else:
                h_compensated = w * height // width
                final_img = np.ones((h_compensated, w), dtype="uint8") * 255
                final_img[int(h_compensated / 2 - h / 2)
                              :int(h_compensated / 2 + h / 2), 0:w] = np_img

        else:
            if h > height:
                w_compensated = h * width // height
                final_img = np.ones((h, w_compensated), dtype="uint8") * 255
                final_img[0:h, int(w_compensated / 2 - w / 2)                          :int(w_compensated / 2 + w / 2)] = np_img
            else:
                final_img = np.ones((height, width), dtype="uint8") * 255
                final_img[int(height / 2 - h / 2):int(height / 2 + h / 2),
                          int(width / 2 - w / 2):int(width / 2 + w / 2)] = np_img

        if final_img.shape != (height, width):
            final_img = np.resize(final_img, (height, width))
        return np.reshape(final_img, (height, width, 1))


path_validation = "E:\\Polytech_Projects\\pfe_plankton_classif\\Dataset\\uvp5ccelter\\"
path_model = "./tree_classifiers/model_cluster2/model_level0_new_hierarchique.h5"
path_dataset = "E:\Polytech_Projects\pfe_plankton_classif\Dataset\DATASET\level0_new_hierarchique22"

benchmark = Benchmark(path_validation)
classifier = TreeClassifier(path_dataset, load_model=True)
# model = load_model(path_model)
benchmark.benchmark_TreeClassifier(classifier)

# classes = ['Annelida', 'Arthropoda', 'Chaetognatha', 'Chordata', 'Cnidaria', 'Ctenophora',
#'Echinodermata', 'Enteropneusta (Hemichordata XX)', 'Thecosomata']

# tt = "E:\\Polytech_Projects\\pfe_plankton_classif\\Dataset\\DATASET\\level0_new_hierarchique22\\Metazoa"
# x, y = benchmark.load_all(tt, classes)
