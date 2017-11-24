#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Augment a ZooProcess dataset with width and height.

The ZooCam dataset file lacks the fields `width` and `height` that are needed
to compare the image sizes of the datasets among each other.

Every image is opened and its dimensions are saved to the output file.

@author: mschroeder
"""

import csv
import sys
from argparse import ArgumentParser
import cv2
import os

def main(argv=None):
    # Setup argument parser
    parser = ArgumentParser(
            description="""Augment a ZooProcess dataset with width and height.""")
    parser.add_argument("source", help="Path to source ZooProcess dataset file")
    parser.add_argument("image_root", help="Path to image root")
    parser.add_argument("dest", help="Path to dest ZooProcess dataset file")
    parser.add_argument("--ext", type=str, default="jpg",
                        help="Filename extension for images.")

    args = parser.parse_args()
    
    print("Arguments:")
    for arg, val in vars(args).items():
        print(" {}: {}".format(arg, val))
    print()
    
    with open(args.source, "r", encoding="utf-8-sig") as f_in, \
        open(args.dest, 'w', encoding='utf-8') as f_out:
         
        reader = csv.DictReader(f_in, delimiter=',')
        
        fieldnames = reader.fieldnames + ["width", "height"]
        writer = csv.DictWriter(f_out, fieldnames, delimiter=',')
        
        writer.writeheader()

        for i, row in enumerate(reader):
            unique_name, objid = row["unique_name"], row["objid"]
            
            # Convert to standard integer notation
            objid = str(int(float(objid)))
            
            # Find source image
            img_basename = objid + "." + args.ext
            source_img_path = os.path.join(
                args.image_root, unique_name, img_basename)
            
            img = cv2.imread(source_img_path)
            
            row["height"] = img.shape[0]
            row["width"] = img.shape[1]
                
            writer.writerow(row)
            
            if i % 100000 == 0 and i > 0:
                print("Processed {:,d} samples.".format(i))

    print("Done.")

if __name__ == "__main__":
    sys.exit(main())
