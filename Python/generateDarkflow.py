import os
import sys
import glob
import random
from fonction_annex import *
from random import gauss
from PIL import Image
from math import sqrt
import numpy as np
from xml.etree.ElementTree import ElementTree, Element, SubElement
import xml.etree.ElementTree as ET

# repository to store the generated images
rep = "../trainingDarkflow/"
# Repositories of the pictures
root = "../Generate/"

# Generate Dir in case they are missing
if not os.path.exists(rep + "images/"):
    os.makedirs(rep + "images/")
if not os.path.exists(rep + "annotations/"):
    os.makedirs(rep + "annotations/")

# backgrounds
rep_background = root + "Backgrounds/"  # backgrounds repository
lst_background = glob.glob(rep_background + "/*.jpg")
nb_back = len(lst_background) # number of backgrounds

# Species:
lst_species = ["acantharia_protist", "ephyra"] # list of species
lst_size = [1,1] # average size of each species
lst_rep = ["acantharia_protist/", "ephyra/"] # repositories
lst_rep_gen = [root+l for l in lst_rep] # list of the repositories for generation


lst_individual = [] # names of the pictures
lst_nb_individual = [] # number of pictures
for i in range(len(lst_species)):
    lst = glob.glob(lst_rep_gen[i] + "/*.png")
    lst_individual.append(lst)
    lst_nb_individual.append(len(lst))

# generate one picture with a certain resolution and of a certain species
def genPic(indiceSpec, indiceImage):
    filename =  str(indiceImage+1) +".jpg"
    back = Image.open(lst_background[random.randint(0, nb_back-1)]).convert("RGBA") # random background
    w, h = back.size
    nbIndividuals = random.randint(1, 2)

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
    widthSize.text = str(w)
    heightSize = SubElement(sizeAnnotation, "height")
    heightSize.text = str(h)
    depthSize = SubElement(sizeAnnotation, "depth")
    depthSize.text = str(3)

    segmentedAnnotation = SubElement(annotation, "segmented")
    segmentedAnnotation.text = str(0)


    # Genenrating a list of random sorted scales
    scaleList = scaleListZooscan(nbIndividuals)

    # paste nbIndividuals Individuals
    for i in range(nbIndividuals):
        nSpec = indiceSpec
        # nSpec = lst_species.index(lst_species[nSpec])
        individual = Image.open(lst_individual[nSpec][random.randint(0,lst_nb_individual[nSpec]-1)]).convert("RGBA") # random Indivdual

        # random modifications
        flipLR, flipTB = random.randint(0, 2), random.randint(0, 2) # random flip
        if (flipLR==0):
            individual = individual.transpose(Image.FLIP_LEFT_RIGHT)
        if (flipTB==0):
            individual = individual.transpose(Image.FLIP_TOP_BOTTOM)
        scale = scaleList[i]
        norm = sqrt(individual.size[1]*individual.size[0]) # norm of the initial cutted out image
        individual = individual.resize([int(lst_size[indiceSpec]*s*scale*150/norm) for s in individual.size], Image.ANTIALIAS)
        individual = individual.rotate(gauss(0,20), expand=1, resample=Image.NEAREST) # random rotation

        # verify that the size of the Individual isn't too big
        if(h-5/6*individual.size[1]<0):
            maxHeight = h*(6/5) - 1
            individual = individual.resize([int(s*maxHeight/individual.size[1]) for s in individual.size], Image.ANTIALIAS)
        if(w-4/6*individual.size[0]<0):
            maxWidth = w*(6/4) -1
            individual = individual.resize([int(s*maxWidth/individual.size[0]) for s in individual.size], Image.ANTIALIAS)

        maxX = round((1/6)*individual.size[0])
        maxY = round((1/6)*individual.size[1])
        posX, posY = random.randint(-1*maxX, w-5*maxX), random.randint(0, h-5*maxY) # random position
        back.paste(individual, (posX, posY), individual)

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
        xmin.text = str(max(0,posX))
        ymin = SubElement(bndbox, "ymin")
        ymin.text = str(max(0,posY))
        xmax = SubElement(bndbox, "xmax")
        xmax.text = str(min(w,posX+windividual))
        ymax = SubElement(bndbox, "ymax")
        ymax.text = str(min(h,posY+hindividual))

    back = back.convert('RGB')
    back = gaussianNoise(back)

    back.save(rep + "images/" + filename, 'JPEG')
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
    numPics = 25 # number of pictures to generate for each class
    genPics(numPics)

main()
print("\n\ndone.")
