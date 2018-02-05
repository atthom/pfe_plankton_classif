# imports
import os
import sys
import glob
import random
from random import uniform
from annexFunctions import *
from PIL import Image
import numpy as np

option_rotation = True
option_changeSize = True
option_noise = True
option_flip = True

height = 250
width = 250

def randomTransform(img):
    s1 , s2 = img.size
    w, h = (int(s1*2.5), int(s2*2.5))
    back = Image.new('RGB', (w, h), (255,255,255)).convert("RGBA")

    if(option_flip):
        flipLR, flipTB = random.randint(0, 2), random.randint(0, 2) # random flip
        if (flipLR==0):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if (flipTB==0):
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

    if(option_changeSize):
        scale = uniform(0.5, 1.5)
        img = img.resize([int(s*scale) for s in img.size], Image.ANTIALIAS)

    if(option_rotation):
        img = img.rotate(uniform(-90, 90), expand=1, resample=Image.NEAREST) # random rotation

    # verify that the size of the img isn't too big
    if(h-5/6*img.size[1]<0):
        maxHeight = h*(6/5) - 1
        img = img.resize([int(s*maxHeight/img.size[1]) for s in img.size], Image.ANTIALIAS)
    if(w-4/6*img.size[0]<0):
        maxWidth = w*(6/4) -1
        img = img.resize([int(s*maxWidth/img.size[0]) for s in img.size], Image.ANTIALIAS)
    maxX = round((1/6)*img.size[0])
    maxY = round((1/6)*img.size[1])
    posX, posY = random.randint(-1*maxX, max(-1*maxX+1,w-5*maxX)), random.randint(-1*maxY, max(-1*maxY+1,h-5*maxY)) # random position
    back.paste(img, (posX, posY), img) # paste the object in a random position of the image

    back = back.convert('RGB')
    if(option_noise):
        back = gaussianNoiseRGB(back,2) # Add a gaussian noise of standard deviation
    back = back.convert('L')
    return back

def processFile(path_file,data_dir, destination, id_img):
    new_path_file = destination + "/" + str(id_img) + ".png"
    img = Image.open(path_file).convert("RGBA")
    topLeftx,topLefty,bottomRightx,bottomRighty = extractMinFrameAlpha(img)
    new_img = img.copy()
    if((bottomRightx - topLeftx > 1) & ( bottomRighty - topLefty > 1) ):
        box = (topLefty,topLeftx,bottomRighty,bottomRightx)
        box2 = extractMinObject(path_file,topLefty,topLeftx,bottomRighty,bottomRightx)
        new_img = new_img.crop(box2)
        new_img = imgWithAlphaProportionalRGBA(new_img)
        new_img = randomTransform(new_img)
        new_img.save(new_path_file, 'png')

        final_img = addBackground(new_path_file,height,width)
        final_img = Image.fromarray(final_img)
        final_img = final_img.resize((width, height), Image.ANTIALIAS)
        final_img.save(new_path_file)

def dataAugmentation(numPics,data_dir,destination):
    dir_names = [_ for _ in os.listdir(data_dir)]
    for dirname in dir_names:
        print(dirname)
        path = data_dir + dirname
        new_path = destination + dirname
        create_dir_if_needed(new_path)
        files = glob.glob(path + "/*.jpg")
        lenfiles = len(files)
        picsPerFile = numPics/lenfiles
        if(numPics<lenfiles):
            for fi in range(numPics):
                print("Picture n°", fi + 1 , "sur", numPics, "with image", fi + 1, "out of ", lenfiles)
                path_file = files[fi]
                processFile(path_file,data_dir, new_path, fi)
        else:
            fi = 0
            for pi in range(numPics):
                print("Picture n°", pi + 1, "sur", numPics, "...", "with image", fi + 1, "out of ", lenfiles)
                path_file = files[fi]
                processFile(path_file,data_dir, new_path, pi)
                if(pi>(fi+1)*picsPerFile):
                    fi = fi + 1


def create_dir_if_needed(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def main():
    rep = "Kaggle_Dataset/"
    data_dir = "D:/Travaille/projet/YOLO/" + rep
    destination = "D:/Travaille/projet/YOLO/Generate/" + "test/"
    numPics = 100
    dataAugmentation(numPics,data_dir,destination)

main()
print("/n/ndone.")
