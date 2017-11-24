#!/usr/bin/env python3
# encoding: utf-8
'''
Extract certain columns from a ZooProcess data file.

@author: Martin Schroeder <sms@informatik.uni-kiel.de>
'''

import sys
from argparse import ArgumentParser
import csv

def main():
    # Setup argument parser
    parser = ArgumentParser(description='''
    Extract columns from a ZooProcess data file.
    ''')
    parser.add_argument(
        "in_",
        help="Path to input data file")
    parser.add_argument(
        "out",
        help="Path to output data file")
    parser.add_argument(
        "column",
        nargs="+",
        help="Column to appear in OUT")
    
    # Process arguments
    args = parser.parse_args()
    
    columns = args.column
    
    print("Arguments:")
    for arg, val in vars(args).items():
        print(" {}: {}".format(arg, val))
    print()

    with open(args.in_, "r", newline='') as f_in, open(args.out, "w", newline='') as f_out:
        reader = csv.DictReader(f_in, delimiter=",")
        
        if not all(c in reader.fieldnames for c in columns):
            raise Exception("IN does not contain the following columns: {}".format(", ".join(c for c in columns if c not in reader.fieldnames)))
        
        writer = csv.DictWriter(f_out, fieldnames=columns, delimiter=",")
        writer.writeheader()

        for i, row in enumerate(reader):
            writer.writerow({k: row[k] for k in columns})

            if i > 0 and i % 100000 == 0:
                print("Processed {:,d} samples.".format(i))


if __name__ == "__main__":
    sys.exit(main())
