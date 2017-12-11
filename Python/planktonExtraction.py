# imports
from PIL import Image
from annexFunctions import *
import numpy as np
import os
import glob

def generate_pictures():
     # replace ("pfe_plankton_classif/LOOV/uvp5ccelter_group1/", "YOLO/Generate/Alpha_uvp5/")
    train_data_dir = "/home/mahiout/Documents/Projets/YOLO/uvp5ccelter_group1/"

    dir_names = [_ for _ in os.listdir(train_data_dir)]
    print("Generating pictures...")
    for dirname in dir_names:
        print(dirname)
        path = train_data_dir + dirname
        create_dir_if_needed(path)
        files = glob.glob(path + "/*.jpg")
        lenfiles = len(files)

        for fi in range(min(500, lenfiles)):
            print("Picture n°", fi, "sur", min(500, lenfiles), "...")
            path_file = files[fi]
            new_path_file = path_file.replace("jpg", "png").replace("uvp5ccelter_group1/", "Generate/Alpha_uvp5v2/")

            img = Image.open(path_file).convert("RGBA")
            topLeftx,topLefty,bottomRightx,bottomRighty = extractMinFrame(img)
            new_img = img.copy()
            if((bottomRightx - topLeftx > 1) & ( bottomRighty - topLefty > 1) ):
                box = (topLefty,topLeftx,bottomRighty,bottomRightx)
                box2 = extractMinObject(path_file,topLefty,topLeftx,bottomRighty,bottomRightx)
                new_img = new_img.crop(box2)
                new_img = imgWithAlphaProportional(new_img)
                new_img.save(new_path_file, 'png')


def create_dir_if_needed(path):
    dd = path.replace("uvp5ccelter_group1/", "Generate/Alpha_uvp5v2/")

    if not os.path.isdir(dd):
        os.makedirs(dd)

def main():
    generate_pictures()

main()
print("\n\ndone.")
