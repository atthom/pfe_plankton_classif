#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create an image collection based on a collection index.

A folder with the same name as the index file is created and populated with
folders of images according to the filename and class in the index file.

@author: mschroeder
"""

import csv
import os
import sys
from argparse import ArgumentParser

def main(argv=None):
    # Setup argument parser
    parser = ArgumentParser(description="""
    Create an image collection based on a collection index.
    """)
    parser.add_argument("collection_index", nargs="+", help="Path to collection index file. ")
    parser.add_argument("--prefix", help="Path to destination.")

    args = parser.parse_args()
    
    print("Arguments:")
    for arg, val in vars(args).items():
        print(" {}: {}".format(arg, val))
    print()

    
    for collection_index in args.collection_index:
        print("Processing {}...".format(collection_index))
        index_dir = os.path.dirname(collection_index)
        
        collection_path = os.path.splitext(collection_index)[0]
        
        if args.prefix:
            collection_path = os.path.join(args.prefix, collection_path)
        
        num_samples = 0
        
        with open(collection_index, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=',')
    
            for i, (rel_source_img_path, effective_name) in enumerate(reader):
                num_samples += 1
                # Find source image
                img_basename = os.path.basename(rel_source_img_path)
                source_img_path = os.path.join(index_dir, rel_source_img_path)
                
                try:
                    dest_dir = os.path.join(collection_path, effective_name)
                    dest_fn = os.path.join(dest_dir, img_basename)
                    os.link(source_img_path, dest_fn)
                except FileNotFoundError:
                    if not os.path.isfile(source_img_path):
                        print("Source image %s not found. Rebuild index!" % source_img_path, file=sys.stderr, flush=True)
                        raise
                    os.makedirs(os.path.abspath(dest_dir))
                    os.link(source_img_path, dest_fn)
                    
                if i % 100000 == 0 and i > 0:
                    print("Processed {:,d} samples.".format(i))
                    
        print("Done processing {:,d} samples.".format(num_samples))
        
if __name__ == "__main__":
    sys.exit(main())
