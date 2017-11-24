#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"Create an index of image files based on a ZooProcess dataset
and potentially prepare for cross-validation.

The resulting index file will contain image paths (relativ to the index file),
class names (according to the selected column in the ZooProcess data file).

For k-fold cross-validation, k pairs of training/validation indices are created.
This requires a `split` column in the ZooProcess dataset.
An object is assigned to the validation split with index i
(and to the training splits with index != i) for i = `split` MOD k.

For a stratified split (which preserves the class proportions of the full dataset),
`split` should be an individual (zero-based) running index of objects for
every category (see split_dataset.py).

@author: mschroeder
"""

import csv
import os
import sys
from argparse import ArgumentParser
from contextlib import ExitStack

def main():
    # Setup argument parser
    parser = ArgumentParser(
            description="""Create an index of image files based on a ZooProcess
            dataset and potentially prepare for cross-validation.""")
    parser.add_argument("dataset", help="Path to ZooProcess dataset file")
    parser.add_argument("image_root", help="Path to image root")
    parser.add_argument("dest_prefix", help="Prefix for destination")
    parser.add_argument(
        "--cross-validation",
        metavar="N",
        type=int,
        help="Create N individual splits for cross-validation.")
    parser.add_argument("--ext", type=str, default="jpg",
                        help="Filename extension for images.")
    parser.add_argument("--effective-name",
                        default = "unique_name",
                        help="Column for effective class name.")

    args = parser.parse_args()
    
    print("Arguments:")
    for arg, val in vars(args).items():
        print(" {}: {}".format(arg, val))
    print()

    ext = args.ext
    
    effective_name_col = args.effective_name
    
    if args.cross_validation:
        n_splits = args.cross_validation
        splits = tuple(range(n_splits))
    else:
        splits = None

    num_samples = 0
    num_not_found = 0
    num_ignored = 0
    
    dest_dir = os.path.dirname(args.dest_prefix)

    with ExitStack() as stack:
        f_in = stack.enter_context(
            open(args.dataset, 'r', encoding='utf-8-sig'))
        reader = csv.DictReader(f_in, delimiter=',')

        if splits:
            if "split" not in reader.fieldnames:
                raise Exception("File {} does not contain column \"split\".".format(args.dataset))
                
            keys = list("%d_%s" % (split, mode)
                        for split in splits for mode in ['train', 'val'])
            f_out = {
                k: stack.enter_context(
                    open(
                        args.dest_prefix +
                        "_" +
                        k +
                        ".csv",
                        'w',
                        encoding='utf-8')) for k in keys}
            writer = {k: csv.writer(f, delimiter=',')
                      for k, f in f_out.items()}
        else:
            f_out = stack.enter_context(
                open(
                    args.dest_prefix +
                    ".csv",
                    'w',
                    encoding='utf-8'))
            writer = csv.writer(f_out, delimiter=',')

        for i, row in enumerate(reader):
            unique_name, objid = row["unique_name"], row["objid"]
            
            effective_name = row[effective_name_col]
            
            if len(effective_name) == 0:
                num_ignored += 1
                continue
            
            # Convert to standard integer notation
            objid = str(int(float(objid)))

            num_samples += 1

            # Find source image
            img_basename = objid + "." + ext
            source_img_path = os.path.join(
                args.image_root, unique_name, img_basename)
            rel_source_img_path = os.path.relpath(source_img_path, dest_dir)

            if not os.path.isfile(source_img_path):
                print("Source image %s not found." % source_img_path)
                num_not_found += 1
                continue

            if splits:
                current_split = int(row["split"]) % n_splits
                for split in splits:
                    collection = str(split) \
                        + ("_val" if current_split == split else "_train")
                    writer[collection].writerow(
                        (rel_source_img_path, effective_name))
            else:
                writer.writerow((rel_source_img_path, effective_name))

            if i % 100000 == 0 and i > 0:
                print("Processed {:,d} samples.".format(i))

    print("Done processing {:,d} samples.".format(num_samples))
    print("{} empty entries for effective name in {} were ignored.".format(num_ignored, effective_name_col))
    
    if num_not_found > 0:
        print("%d samples (%.2f%%) not found." %
              (num_not_found, num_not_found / num_samples * 100))


if __name__ == "__main__":
    sys.exit(main())
