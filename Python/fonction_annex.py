from PIL import Image
import numpy as np
import os
import cv2
import random

# Generate a gaussian noise on a given image
def gaussianNoise(image):
    row,col = image.size
    image =  np.asarray(image)
    mean = 0
    sigma = 4

    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.astype(int)
    gauss = gauss.reshape(col,row)

    newImage = image.copy()

    newImage[:,:,0] = np.clip(image[:,:,0] + gauss, 0, 255)
    newImage[:,:,1] = np.clip(image[:,:,1] + gauss, 0, 255)
    newImage[:,:,2] = np.clip(image[:,:,2] + gauss, 0, 255)

    newImage = Image.fromarray(newImage, 'RGB')
    return newImage

# Genenrating a list of random sorted scales
def scaleListZooscan(nbIndividuals):
    scaleList = []
    for i in range(nbIndividuals):
        d = random.randint(0, 10)  # random uniform scale for the position of the Individual
        scale = 30/(20+d) # perceived scale
        scaleList.append(scale)
    # Sorting the list of scale in order to put the biggest Individual in front of the new image
    scaleList.sort()
    return scaleList
