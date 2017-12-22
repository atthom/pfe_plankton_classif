# imports
from PIL import Image
from annexFunctions import *
import numpy as np
import os
import glob

def generate_pictures(train_data_dir,rep):
    dir_names = [_ for _ in os.listdir(train_data_dir)]
    print("Generating pictures...")
    for dirname in dir_names:
        print(dirname)
        path = train_data_dir + dirname
        new_path = path.replace(rep, "Generate/" + rep)
        create_dir_if_needed(new_path)
        files = glob.glob(path + "/*.jpg")
        lenfiles = len(files)
        totalArea = 0
        notEmptyFrame = 0

        for fi in range(min(500, lenfiles)):
            print("Picture n°", fi, "sur", min(500, lenfiles), "...")
            path_file = files[fi]
            new_path_file = path_file.replace("jpg", "png").replace(rep, "Generate/" + rep)

            img = Image.open(path_file).convert("RGBA")
            topLeftx,topLefty,bottomRightx,bottomRighty = extractMinFrame(img)
            new_img = img.copy()
            if((bottomRightx - topLeftx > 1) & ( bottomRighty - topLefty > 1) ):
                box = (topLefty,topLeftx,bottomRighty,bottomRightx)
                box2 = extractMinObject(path_file,topLefty,topLeftx,bottomRighty,bottomRightx)
                new_img = new_img.crop(box2)
                new_img = imgWithAlphaProportional(new_img)
                new_img.save(new_path_file, 'png')

                totalArea = totalArea + (bottomRightx - topLeftx)*( bottomRighty - topLefty)
                notEmptyFrame = notEmptyFrame + 1


        averageArea = totalArea/notEmptyFrame
        path_area_file = new_path + "/area.txt"
        if os.path.isfile(path_area_file):
            print("--- remove old version ---")
            os.remove(path_area_file)
        area_file = open(path_area_file, "a")
        area_file.write(str(averageArea))
        area_file.close()

def create_dir_if_needed(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def main():
    rep = "Kaggle_Dataset/"
    train_data_dir = "/home/mahiout/Documents/Projets/YOLO/" + rep
    destination = "/home/mahiout/Documents/Projets/YOLO/Generate/" + rep
    generate_pictures(train_data_dir,rep)


main()
print("\n\ndone.")
