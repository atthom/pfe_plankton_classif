# imports
from math import sqrt
import numpy as np
import os
import glob
import random
import shutil

def splitByArea(train_data_dir,destination,k):
    dir_names = [_ for _ in os.listdir(train_data_dir)]
    areas = []
    print("Proccessing folders...")
    for i in range(len(dir_names)):
        dirname = dir_names[i]
        print(dirname)
        path_area_file = train_data_dir + dirname + "/area.txt"

        area_file = open(path_area_file, "r")
        area = area_file.read()
        areas.append(sqrt(float(area)))
        area_file.close()

    mu = find_centers(areas, k)
    for j in range(len(dir_names)):
        dirname = dir_names[j]
        bestmukey = min([(i[0], np.linalg.norm(areas[j]-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        print(mu[bestmukey])
        print(dirname)
        path_destination_files = destination + "/" + str(mu[bestmukey]) + "/" + dirname
        path_files = train_data_dir + dirname
        create_dir_if_needed(path_destination_files)

        for fic in os.listdir(path_files):
            shutil.copy2((path_files + "/" + fic),path_destination_files)

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters

def has_converged(mu, oldmu):
    return (sorted(mu) == sorted(oldmu))

def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return mu

def create_dir_if_needed(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def main():
    rep = "Kaggle_Dataset/"
    train_data_dir = "/home/mahiout/Documents/Projets/YOLO/Generate/" + rep
    destination = "/home/mahiout/Documents/Projets/YOLO/Generate/Kaggle_Dataset_Split"
    splitByArea(train_data_dir,destination,4)



main()
print("\n\ndone.")
