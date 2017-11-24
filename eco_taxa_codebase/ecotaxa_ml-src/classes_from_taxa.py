#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract a list of classes from a taxa mapping.

@author: mschroeder
"""

import csv
import sys
from argparse import ArgumentParser

import numpy as np


def main(argv=None):
    # Setup argument parser
    parser = ArgumentParser(
        description="""Extract a list of classes from a taxa mapping.""",
        epilog="""Mapping: MAPPING_FILE is a CSV file with a column unique_name.
               An additional column specified by COLUMN contains the new class name.""")
    parser.add_argument(
        "mapping_file",
        help="Filename for class name mapping.")
    parser.add_argument("column", help="Column for class name mapping.")
    parser.add_argument(
        "classes_file",
        help="Path to classes file")

    args = parser.parse_args()
    
    print("Arguments:")
    for arg, val in vars(args).items():
        print(" {}: {}".format(arg, val))
    print()

    with open(args.mapping_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=',')

        classes = set(row[args.column] for row in reader if row[args.column])

    classes = sorted(classes)

    print("{} classes in column {}:".format(len(classes), args.column), ", ".join(classes))

    np.savetxt(args.classes_file, classes, fmt="%s")


if __name__ == "__main__":
    sys.exit(main())
