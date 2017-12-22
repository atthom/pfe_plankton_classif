from keras.models import save_model, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import csv
import json
import PIL
import os
import random
import numpy as np
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
        
        first = tree_classifier.directories[0]
        model = tree_classifier.models[first]
        label = tree_classifier.labels[first]
        
        x, y = self.load_all(label)
        y_pred = model.predict(x, batch_size=50)
        print(len(y_pred))
        self.heatmap(y)
        self.heatmap(y_pred)
        

    def heatmap(self, matrix):
        import matplotlib.pyplot as plt
        plt.imshow(matrix, cmap='hot', interpolation='nearest')
        plt.show()

    def load_all(self, classes):
        keys = list(self.database.keys())
        print("Loading", len(keys), "images...")
        random.shuffle(keys)
        keys = keys[0:50]
        x = []
        y = []

        for path in keys:
            base_anwser = np.asarray([0] * len(classes))
            i = 0
            for p in classes:
                print(path, self.database[path], p)
                if p in path.split(os.sep):
                    base_anwser[i] = 1.
                i += 1
            img = self.load_image(self.path_img + os.sep + path)
            x.append(img)
            y.append(base_anwser)

            if len(x) % 1000 == 0:
                print(len(x), "images loaded...")

        return np.array(x), np.array(y)

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
                if h/height > w/width:
                    w_compensated = h * width // height
                    final_img = np.ones((h,w_compensated), dtype="uint8") * 255
                    final_img[0:h,int(w_compensated/2 - w/2):int(w_compensated/2 + w/2)] = np_img
                else :
                    h_compensated = w * height // width
                    final_img = np.ones((h_compensated, w), dtype="uint8") * 255
                    final_img[int(h_compensated/2 - h/2):int(h_compensated/2 + h/2), 0:w] = np_img
            else :
                h_compensated = w * height // width
                final_img = np.ones((h_compensated, w), dtype="uint8") * 255
                final_img[int(h_compensated/2 - h/2):int(h_compensated/2 + h/2), 0:w] = np_img
            
        else:
            if h > height:
                w_compensated = h * width // height
                final_img = np.ones((h, w_compensated), dtype="uint8") * 255
                final_img[0:h,int(w_compensated/2 - w/2):int(w_compensated/2 + w/2)] = np_img
            else:
                final_img = np.ones((height, width), dtype="uint8") * 255
                final_img[int(height/2 - h/2):int(height/2+h/2), int(width/2 - w/2):int(width/2 + w/2)] = np_img
        
        if final_img.shape != (height, width):
            final_img = np.resize(final_img, (height, width))
        return np.reshape(final_img, (height, width, 1))


path_validation = "E:\\Polytech_Projects\\pfe_plankton_classif\\Dataset\\uvp5ccelter\\"
path_model = "./tree_classifiers/model_cluster2/model_level0_new_hierarchique.h5"
path_dataset = "E:\Polytech_Projects\pfe_plankton_classif\Dataset\DATASET\level0_new_hierarchique22"

benchmark = Benchmark(path_validation)
classifier = TreeClassifier(path_dataset, load_model=True)
#model = load_model(path_model)
benchmark.benchmark_TreeClassifier(classifier)
