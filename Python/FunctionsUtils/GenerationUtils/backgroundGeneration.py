# imports
from PIL import Image
from numpy import *

def backgrounfGenerationRGB(width,height,path):
    size = (width,height)
    img = Image.new('RGB', size, (255,255,255))
    img.save(path, 'png')

def  backgrounfGeneration(width,height,path):
    size = (width,height)
    img = Image.new('L', size, (255))
    img.save(path, 'png')

def main():
    path = "background.png"
    path2 = "backgroundRGB.png"
    width = 400
    height = 400
    backgrounfGeneration(width,height,path)
    backgrounfGenerationRGB(width,height,path2)

main()
print("\n\ndone.")
