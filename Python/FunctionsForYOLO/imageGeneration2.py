# imports
import os
import sys
import glob
import random
from random import uniform

sys.path.append('..')
from FunctionsUtils.GenerationUtils.annexFunctions import *

from PIL import Image
from math import sqrt
import numpy as np
from xml.etree.ElementTree import ElementTree, Element, SubElement
import xml.etree.ElementTree as ET

# repository to store the generated images
rep = "D:/Travaille/projet/YOLO/trainingDarkflow/"
# Repositories of the pictures
root = "D:/Travaille/projet/YOLO/uvp5ccelter_group1/"

heightProcessed = 150
widthProcessed = 150

# Generate Dir in case they are missing
if not os.path.exists(rep + "images/"):
    os.makedirs(rep + "images/")
if not os.path.exists(rep + "annotations/"):
    os.makedirs(rep + "annotations/")


# Species:
lst_species = [_ for _ in os.listdir(root)]
for species in lst_species:
    print(species)
lst_rep_gen = [root + l for l in lst_species ] # list of the repositories for generation


lst_individual = [] # names of the pictures
lst_nb_individual = [] # number of pictures
for i in range(len(lst_species)):
    lst = glob.glob(lst_rep_gen[i] + "/*.jpg")
    lst_individual.append(lst)
    lst_nb_individual.append(len(lst))
    print(len(lst))

# generate one picture with a certain resolution and of a certain species
def genPic(indiceSpec, indiceImage):
    filename =  str(indiceImage+1) +".jpg"

    scaleList = scaleListZooscan(1)
    nSpec = indiceSpec
    # nSpec = lst_species.index(lst_species[nSpec])
    pathImage = lst_individual[nSpec][random.randint(0,lst_nb_individual[nSpec]-1)]
    individual = Image.open(pathImage).convert("RGBA") # random Indivdual

    # random modifications
    flipLR, flipTB = random.randint(0, 2), random.randint(0, 2) # random flip
    if (flipLR==0):
        individual = individual.transpose(Image.FLIP_LEFT_RIGHT)
    if (flipTB==0):
        individual = individual.transpose(Image.FLIP_TOP_BOTTOM)
    scale = scaleList[0]
    individual = individual.resize([int(s*scale) for s in individual.size], Image.ANTIALIAS)
    individual = individual.rotate(uniform(0, 12.5), expand=1, resample=Image.NEAREST) # random rotation
    white = Image.new('RGBA', individual.size, (255,)*4)
    # create a composite image using the alpha layer of rot as a mask
    individual = Image.composite(individual, white, individual)
    w, h = individual.size

    # Creating the xml file
    annotation = Element('annotation')
    tree = ElementTree(annotation)

    folderAnnotation = SubElement(annotation, "folder")
    folderAnnotation.text = 'images/trainingDarkflow/'

    filenameAnnotation = SubElement(annotation, "filename")
    filenameAnnotation.text = str(indiceImage+1) +".jpg"

    sourceAnnotation = SubElement(annotation, "source")
    databaseSource = SubElement(sourceAnnotation, "database")
    databaseSource.text = 'trainingDarkflow'
    annotationSource = SubElement(sourceAnnotation, "annotation")
    annotationSource.text = '...'
    imageSource = SubElement(sourceAnnotation, "image")
    imageSource.text = '...'
    flickridSource = SubElement(sourceAnnotation, "flickrid")
    flickridSource.text = '...'

    ownerAnnotation = SubElement(annotation, "owner")
    flickridOwner = SubElement(ownerAnnotation, "flickrid")
    flickridOwner.text = 'Polytech'
    nameOwner = SubElement(ownerAnnotation, "name")
    nameOwner.text = 'unknown'

    sizeAnnotation = SubElement(annotation, "size")
    widthSize = SubElement(sizeAnnotation, "width")
    widthSize.text = str(widthProcessed)
    heightSize = SubElement(sizeAnnotation, "height")
    heightSize.text = str(heightProcessed)
    depthSize = SubElement(sizeAnnotation, "depth")
    depthSize.text = str(3)

    segmentedAnnotation = SubElement(annotation, "segmented")
    segmentedAnnotation.text = str(0)

    individual = individual.convert('RGB')
    individual.save(rep + "images/" + filename, 'JPEG')
    final_img = addBackground(rep + "images/" + filename,heightProcessed,widthProcessed)
    final_img = Image.fromarray(final_img)
    final_img = final_img.resize((widthProcessed, heightProcessed), Image.ANTIALIAS)
    final_img.save(rep + "images/" + filename, 'png')

    topLeftx,topLefty,bottomRightx,bottomRighty = extractMinFrameAlpha(final_img.convert('RGBA'))
    if((bottomRightx - topLeftx > 1) & ( bottomRighty - topLefty > 1) ):
        box = extractMinObject(rep + "images/" + filename,topLefty,topLeftx,bottomRighty,bottomRightx)
        topLefty = box[0];topLeftx = box[1];bottomRighty = box[2];bottomRightx = box[3]

        # Writing the Individual specifications on the xml file
        objectAnnotation = SubElement(annotation, "object")
        nameObject = SubElement(objectAnnotation, "name")
        nameObject.text = lst_species[indiceSpec]
        poseObject = SubElement(objectAnnotation, "pose")
        poseObject.text = 'Left'
        truncatedObject = SubElement(objectAnnotation, "truncated")
        truncatedObject.text = str(1)
        difficultObject = SubElement(objectAnnotation, "difficult")
        difficultObject.text = str(0)

        bndbox = SubElement(objectAnnotation, "bndbox")
        windividual, hindividual = individual.size
        xmin = SubElement(bndbox, "xmin")
        xmin.text = str(topLeftx)
        ymin = SubElement(bndbox, "ymin")
        ymin.text = str(topLefty)
        xmax = SubElement(bndbox, "xmax")
        xmax.text = str(bottomRightx)
        ymax = SubElement(bndbox, "ymax")
        ymax.text = str(bottomRighty)

    tree.write(open( rep + "annotations/" + str(indiceImage+1) + ".xml", 'wb'))


def displayLoading(percent):
    nbcar = 10
    pas = 100./nbcar
    s = '\r['
    for i in range(1,nbcar+1):
        if i <= percent//pas:
            s += '#'
        else:
            s += '-'
    s += ']'
    sys.stdout.write(s)

# generate numPics pictures with a certain resolution
def genPics(numPics):
    for j in range(len(lst_species)):
        print("\n" + str(lst_species[j]))
        displayLoading(0)
        for i in range(numPics):
            genPic(j,i+j*numPics)
            displayLoading(100*i/numPics)
        displayLoading(100)

def main():
    numPics = 2500 # number of pictures to generate for each class
    genPics(numPics)

main()
print("\n\ndone.")
