# imports
import os
import sys
import glob
import random
from PIL import Image

# Taking argument into account to find out the repository to be processed
if len(sys.argv) >= 2:
    imageDir = sys.argv[1]
    testingDir = sys.argv[2]
else:
    print("Python need the path of the repository containing the images for sampling")
    sys.exit(0)

def sampling(imageDir,testingDir,sample):
    dir_names_level1 = [_ for _ in os.listdir(imageDir)]
    for dirname_level1 in dir_names_level1:
        print(dirname_level1)
        pathImage_level1 = imageDir + "/" + dirname_level1
        pathTest_level1 = testingDir + "/" + dirname_level1
        create_dir_if_needed(pathTest_level1)
        files = glob.glob(pathImage_level1 + "/*.jpg")
        lenfiles = len(files)
        if (lenfiles==0):
            dir_names_level2 = [_ for _ in os.listdir(pathImage_level1)]
            for dirname_level2 in dir_names_level2:
                print(dirname_level2)
                pathImage_level2 = pathImage_level1 + "/" + dirname_level2
                pathTest_level2 = pathTest_level1 + "/" + dirname_level2
                create_dir_if_needed(pathTest_level2)
                files = glob.glob(pathImage_level2 + "/*.jpg")
                lenfiles = len(files)
                for fi in range(lenfiles):
                    print("Picture n°", fi, "sur", lenfiles, "...")
                    path_file = files[fi]
                    proba = random.random()
                    if ( proba <= sample ):
                        new_path_file = path_file.replace(pathImage_level2, pathTest_level2)
                        img = Image.open(path_file)
                        new_img = img.copy()
                        print(new_path_file)
                        new_img.save(new_path_file, 'JPEG')
                        os.remove(path_file)

        else:
            for fi in range(lenfiles):
                print("Picture n°", fi, "sur", lenfiles, "...")
                path_file = files[fi]
                proba = random.random()
                if ( proba <= sample ):
                    new_path_file = path_file.replace(pathImage_level1, pathTest_level1)
                    img = Image.open(path_file)
                    new_img = img.copy()
                    print(new_path_file)
                    new_img.save(new_path_file, 'JPEG')
                    os.remove(path_file)

def samplingLimited(imageDir,testingDir,limite):
    dir_names_level1 = [_ for _ in os.listdir(imageDir)]
    for dirname_level1 in dir_names_level1:
        print(dirname_level1)
        pathImage_level1 = imageDir + "/" + dirname_level1
        pathTest_level1 = testingDir + "/" + dirname_level1
        create_dir_if_needed(pathTest_level1)
        files = glob.glob(pathImage_level1 + "/*.jpg")
        lenfiles = len(files)
        if (lenfiles==0):
            dir_names_level2 = [_ for _ in os.listdir(pathImage_level1)]
            for dirname_level2 in dir_names_level2:
                print(dirname_level2)
                pathImage_level2 = pathImage_level1 + "/" + dirname_level2
                pathTest_level2 = pathTest_level1 + "/" + dirname_level2
                create_dir_if_needed(pathTest_level2)
                files = glob.glob(pathImage_level2 + "/*.jpg")
                lenfiles = len(files)
                for fi in range(min(limite,lenfiles)):
                    print("Picture n°", fi, "sur", lenfiles, "...")
                    path_file = files[fi]
                    new_path_file = path_file.replace(pathImage_level2, pathTest_level2)
                    img = Image.open(path_file)
                    new_img = img.copy()
                    print(new_path_file)
                    new_img.save(new_path_file, 'JPEG')


        else:
            for fi in range(min(limite,lenfiles)):
                print("Picture n°", fi, "sur", lenfiles, "...")
                path_file = files[fi]
                new_path_file = path_file.replace(pathImage_level1, pathTest_level1)
                img = Image.open(path_file)
                new_img = img.copy()
                print(new_path_file)
                new_img.save(new_path_file, 'JPEG')


def reverse(imageDir,testingDir):
    dir_names_level1 = [_ for _ in os.listdir(testingDir)]
    for dirname_level1 in dir_names_level1:
        print(dirname_level1)
        pathImage_level1 = imageDir + "/" + dirname_level1
        pathTest_level1 = testingDir + "/" + dirname_level1
        files = glob.glob(pathTest_level1 + "/*.jpg")
        lenfiles = len(files)
        if (lenfiles==0):
            dir_names_level2 = [_ for _ in os.listdir(pathImage_level1)]
            for dirname_level2 in dir_names_level2:
                print(dirname_level2)
                pathImage_level2 = pathImage_level1 + "/" + dirname_level2
                pathTest_level2 = pathTest_level1 + "/" + dirname_level2
                files = glob.glob(pathTest_level2 + "/*.jpg")
                lenfiles = len(files)
                for fi in range(lenfiles):
                    print("Picture n°", fi, "sur", lenfiles, "...")
                    path_file = files[fi]
                    new_path_file = path_file.replace(pathTest_level2, pathImage_level2)
                    img = Image.open(path_file)
                    new_img = img.copy()
                    new_img.save(new_path_file, 'JPEG')

        else:
            for fi in range(lenfiles):
                print("Picture n°", fi, "sur", lenfiles, "...")
                path_file = files[fi]
                new_path_file = path_file.replace(pathTest_level1, pathImage_level1)
                img = Image.open(path_file)
                new_img = img.copy()
                new_img.save(new_path_file, 'JPEG')


def create_dir_if_needed(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def main():
    samplingLimited(imageDir,testingDir,150);
    #sampling(imageDir,testingDir,0.2);
    #reverse(imageDir,testingDir);

main()
print("\n\ndone.")
