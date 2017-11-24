#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a collection index (that is used in the experiments)
to the new "objid,path,label" format (that is used for SparseConvNet).

@author: mschroeder
"""

import csv
import os
import sys
from argparse import ArgumentParser

def main(argv=None):
    # Setup argument parser
    parser = ArgumentParser(description="""
    Convert a collection index to the new "objid,path,label" format.
    """)
    parser.add_argument("collection_index", help="Path to collection index file.")
    parser.add_argument("ecotaxa_index", help="Path to collection index file.")

    args = parser.parse_args()
    
    print("Arguments:")
    for arg, val in vars(args).items():
        print(" {}: {}".format(arg, val))
    print()

    
    index_dir = os.path.dirname(args.collection_index)
    index_dir = os.path.abspath(index_dir)
    
    with open(args.collection_index, 'r', encoding='utf-8-sig') as f_in, \
        open(args.ecotaxa_index, "w", encoding="utf-8") as f_out:
        reader = csv.reader(f_in, delimiter=',')
        writer = csv.writer(f_out, delimiter=',')

        for i, (rel_source_img_path, effective_name) in enumerate(reader):
            # Find source image
            abs_img_path = os.path.join(index_dir, rel_source_img_path)
            
            objid = os.path.splitext(os.path.basename(rel_source_img_path))[0]
            
            writer.writerow([objid, abs_img_path, effective_name])
                
            if i % 100000 == 0 and i > 0:
                print("Processed {:,d} samples.".format(i))
        
if __name__ == "__main__":
    sys.exit(main())
