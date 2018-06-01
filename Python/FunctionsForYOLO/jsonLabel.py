# imports
import os
import sys
import glob
import json
import numpy as np
from PIL import Image
sys.path.append('..')
from FunctionsUtils.GenerationUtils.annexFunctions import *

# Taking argument into account to find out the repository to be processed
if len(sys.argv) >= 1:
    repository = sys.argv[1]
else:
    print("Python need the path of the repository containing the images to be labelised")
    sys.exit(0)

# Merge the information of right and left parsed sequence
def generateLabel(repository):
    dir_names = [_ for _ in os.listdir(repository)]

    for dirname in dir_names:
        Dir = repository + dirname
        liste = sorted(glob.glob(Dir + "/[0-9]*.jpg"))
        liste = [l.split('/')[-1] for l in liste]

        for l in liste:
            img = Image.open(Dir+"/"+l)
            topLeftx,topLefty,bottomRightx,bottomRighty = extractMinFrame(img)
            if((bottomRightx - topLeftx > 1) & ( bottomRighty - topLefty > 1) ):
                box = extractMinObject(Dir+"/"+l,topLefty,topLeftx,bottomRighty,bottomRightx)
                topLefty = box[0];topLeftx = box[1];bottomRighty = box[2];bottomRightx = box[3]
                data = [{
                    'label' : dirname,
                    'topleft' : {"x": topLeftx, "y": topLefty},
                    'bottomright' : {"x": bottomRightx, "y": bottomRighty}
                }]
                fileName = Dir + "/" + l.split('.')[0] + ".json"
                with open(fileName, 'w') as outfile:
                    json.dump(data,outfile)

def main():
    generateLabel(repository);

main()
print("\n\ndone.")
