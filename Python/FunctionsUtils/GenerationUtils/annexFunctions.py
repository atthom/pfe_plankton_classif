# imports
from PIL import Image
import numpy as np
import os
import cv2
import random
from scipy.misc import imread,imsave

thresholdMax = 250
thresholdMin = 125

# Generate a gaussian noise on a given RGB image.
def gaussianNoiseRGB(image,sigma):
    row,col = image.size
    image =  np.asarray(image)
    mean = 0

    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.astype(int)
    gauss = gauss.reshape(col,row)

    newImage = image.copy()

    newImage[:,:,0] = np.clip(image[:,:,0] + gauss, 0, 255)
    newImage[:,:,1] = np.clip(image[:,:,1] + gauss, 0, 255)
    newImage[:,:,2] = np.clip(image[:,:,2] + gauss, 0, 255)

    newImage = Image.fromarray(newImage, 'RGB')
    return newImage

# Generate a gaussian noise on a given Greyscale image.
def gaussianNoise(image,sigma):
    row,col = image.size
    image =  np.asarray(image)
    mean = 0

    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.astype(int)
    gauss = gauss.reshape(col,row)

    newImage = image.copy()

    newImage[:,:] = np.clip(image[:,:] + gauss, 0, 255)

    newImage = Image.fromarray(newImage, 'L')
    return newImage

# Genenrating a list of random sorted scales
def scaleListZooscan(nbIndividuals):
    scaleList = []
    for i in range(nbIndividuals):
        d = random.randint(0, 10)
        scale = 25/(20+d)
        scaleList.append(scale)
    scaleList.sort()
    return scaleList

# Extract the minimum frame containing all pixels with intensity lower than thresholdMax in an image with transparency.
def extractMinFrameAlpha(img):
    np_img = np.array(img.copy())
    height = np_img.shape[0];
    width = np_img.shape[1];

    topLeftx = height;
    j = 0;
    while((j<width) & (topLeftx !=0)):
        for i in range(topLeftx):
            val = np_img[i,j,0];
            if( (val<=thresholdMax) & (i<topLeftx)):
                topLeftx = i;
        j = j + 1;

    topLefty = width;
    i = 0;
    while((i<height) & (topLefty !=0)):
        for j in range(topLefty):
            val = np_img[i,j,0];
            if( (val<=thresholdMax) & (j<topLefty)):
                topLefty = j;
        i = i + 1;

    bottomRightx = topLeftx;
    j = topLefty;
    while((j<width) & (bottomRightx != width-1)):
        for i in range(height - bottomRightx):
            val = np_img[height - i - 1,j,0];
            if( (val<=thresholdMax) & (height - i - 1 > bottomRightx)):
                bottomRightx = height - i - 1;
        j = j + 1;

    bottomRighty = topLefty;
    i = topLeftx;
    while((i<height) & (bottomRighty != height-1)):
        for j in range(width - bottomRighty):
            val = np_img[i, width - j - 1,0];
            if( (val<=thresholdMax) & (width - j - 1 > bottomRighty)):
                bottomRighty = width - j - 1;
        i = i + 1;
    return topLeftx,topLefty,bottomRightx,bottomRighty

# Extract the minimum frame containing all pixels with intensity lower than thresholdMax in a greyscale image.
def extractMinFrame(img):
    np_img = np.array(img.copy())
    height = np_img.shape[0];
    width = np_img.shape[1];

    topLeftx = height;
    j = 0;
    while((j<width) & (topLeftx !=0)):
        for i in range(topLeftx):
            val = np_img[i,j];
            if( (val<=thresholdMax) & (i<topLeftx)):
                topLeftx = i;
        j = j + 1;

    topLefty = width;
    i = 0;
    while((i<height) & (topLefty !=0)):
        for j in range(topLefty):
            val = np_img[i,j];
            if( (val<=thresholdMax) & (j<topLefty)):
                topLefty = j;
        i = i + 1;

    bottomRightx = topLeftx;
    j = topLefty;
    while((j<width) & (bottomRightx != width-1)):
        for i in range(height - bottomRightx):
            val = np_img[height - i - 1,j];
            if( (val<=thresholdMax) & (height - i - 1 > bottomRightx)):
                bottomRightx = height - i - 1;
        j = j + 1;

    bottomRighty = topLefty;
    i = topLeftx;
    while((i<height) & (bottomRighty != height-1)):
        for j in range(width - bottomRighty):
            val = np_img[i, width - j - 1];
            if( (val<=thresholdMax) & (width - j - 1 > bottomRighty)):
                bottomRighty = width - j - 1;
        i = i + 1;
    return topLeftx,topLefty,bottomRightx,bottomRighty

# Add transparency for the pixel with an intensity superior to thresholdMax
# and apply a linear function of transparency for the pixel with an intensity between thresholdMin and thresholdMax
def imgWithAlphaProportionalRGBA(img):
    np_img = np.array(img.copy())

    for i in range(np_img.shape[0]):
        for j in range(np_img.shape[1]):
            val = np_img[i,j,0];
            if val > thresholdMax:
                np_img[i,j,3] = 0;
            if ((thresholdMin < val) & (val <= thresholdMax )):
                np_img[i,j,3] = - (255/(thresholdMax-thresholdMin))*val + 255*thresholdMax/(thresholdMax-thresholdMin);
    return Image.fromarray(np_img, "RGBA")

# Add transparency for the pixel with an intensity superior to thresholdMax
def imgWithAlphaRGBA(img):
    np_img = np.array(img.copy())

    for i in range(np_img.shape[0]):
        for j in range(np_img.shape[1]):
            val0 = np_img[i,j,0];
            val1 = np_img[i,j,1]
            val2 = np_img[i,j,2]
            if (val0 > thresholdMax)&(val1 > thresholdMax)&(val2 > thresholdMax):
                np_img[i,j,3] = 0;
    return Image.fromarray(np_img, "RGBA")

# Add transparency for the pixel with an intensity superior to thresholdMax
# and apply a linear function of transparency for the pixel with an intensity between thresholdMin and thresholdMax
def imgWithAlphaProportionalLA(img):
    np_img = np.array(img.copy())

    for i in range(np_img.shape[0]):
        for j in range(np_img.shape[1]):
            val = np_img[i,j,0];
            if val > thresholdMax:
                np_img[i,j,1] = 0;
            if ((thresholdMin < val) & (val <= thresholdMax )):
                np_img[i,j,1] = - (255/(thresholdMax-thresholdMin))*val + 255*thresholdMax/(thresholdMax-thresholdMin);
    return Image.fromarray(np_img, "LA")

# Add transparency for the pixel with an intensity superior to thresholdMax
def imgWithAlphaLA(img):
    np_img = np.array(img.copy())

    for i in range(np_img.shape[0]):
        for j in range(np_img.shape[1]):
            val0 = np_img[i,j,0];
            if (val0 > thresholdMax):
                np_img[i,j,1] = 0;
    return Image.fromarray(np_img, "LA")

# Extract the box containing the object with the greatest area in the image.
def extractMinObject(path_file,topLefty,topLeftx,bottomRighty,bottomRightx):
    imgray = cv2.imread(path_file)
    imgray = imgray[topLeftx:bottomRightx, topLefty:bottomRighty]
    imgray = cv2.cvtColor(imgray,cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(imgray,thresholdMax,255,0)
    thresh = (255-thresh)

    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    best = 0
    maxsize = 0
    for cnt in range(len(contours)):
        if cv2.contourArea(contours[cnt]) > maxsize :
            maxsize = cv2.contourArea(contours[cnt])
            best = cnt

    x,y,w,h = cv2.boundingRect(contours[best])
    box = (topLefty+x,topLeftx+y,topLefty+x+w,topLeftx+y+h)
    return box

# Resize an image of plankton without distorting it by filling the gaps with a white background.
def addBackground(path_file,height,width):
    img = Image.open(path_file).convert('L')
    np_img = np.array(img.copy())
    h,w = np_img.shape
    final_img = np.ones((1,1),dtype="uint8")*255
    if(w > width):
        if(h > height):
            if(h/height > w/width):
                w_compensated = int(h*(width/height))
                final_img = np.ones((h,w_compensated),dtype="uint8")*255
                final_img[0:h,int(w_compensated/2-w/2):int(w_compensated/2+w/2)] = np_img
            else :
                h_compensated = int(w*(height/width))
                final_img = np.ones((h_compensated,w),dtype="uint8")*255
                final_img[int(h_compensated/2-h/2):int(h_compensated/2+h/2),0:w] = np_img
        else :
            h_compensated = int(w*(height/width))
            final_img = np.ones((h_compensated,w),dtype="uint8")*255
            final_img[int(h_compensated/2-h/2):int(h_compensated/2+h/2),0:w] = np_img
    else :
        if(h > height):
            w_compensated = int(h*(width/height))
            final_img = np.ones((h,w_compensated),dtype="uint8")*255
            final_img[0:h,int(w_compensated/2-w/2):int(w_compensated/2+w/2)] = np_img
        else :
            final_img = np.ones((height,width),dtype="uint8")*255
            final_img[int(height/2-h/2):int(height/2+h/2),int(width/2-w/2):int(width/2+w/2)] = np_img
    return final_img

'''
def main():
    height = 150
    width = 150
    path_file = "3752.JPG"
    new_path_file = "corrected.png"
    new_path_file2 = "noncorrected.png"
    img = Image.open(path_file).convert("RGBA")
    topLeftx,topLefty,bottomRightx,bottomRighty = extractMinFrame(img)
    new_img = img.copy()
    new_img2 = img.copy()
    if((bottomRightx - topLeftx > 1) & ( bottomRighty - topLefty > 1) ):
        box = (topLefty,topLeftx,bottomRighty,bottomRightx)
        box2 = extractMinObject(path_file,topLefty,topLeftx,bottomRighty,bottomRightx)
        new_img = new_img.crop(box2)
        new_img = new_img.convert('L')
        new_img.save(new_path_file, 'png')
        final_img = addBackground(new_path_file,height,width)
        final_img = Image.fromarray(final_img)
        final_img = final_img.resize((width, height), Image.ANTIALIAS)
        final_img.save(new_path_file)
        new_img2 = new_img2.resize((width, height), Image.ANTIALIAS)
        new_img2.save(new_path_file2, 'png')

main()
'''
