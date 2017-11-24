#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Augment a ZooProcess dataset with columns `name_*` containing the object label
name according to a class mapping specified in MAPPING_FILE.

Mapping: MAPPING_FILE is a CSV file with the column `unique_name`
(specifying the original object label) and at least an additional column
(specifying a new label that should be used instead).

This can be used to group several original object labels to a new general term
or to exclude certain object labels from the experiments.

@author: mschroeder
"""

import csv
import sys
from argparse import ArgumentParser
from fnmatch import fnmatch


def main(argv=None):
    # Setup argument parser
    parser = ArgumentParser(
            description="""Augment a ZooProcess dataset with columns name_* containing
            the name according to a class mapping specified in MAPPING_FILE.""",
            epilog="""      Mapping: MAPPING_FILE is a CSV file with a column unique_name. An additional 
                            column specified by COLUMN contains the new class name.
                            """)
    parser.add_argument("source", help="Path to source ZooProcess dataset file")
    parser.add_argument("dest", help="Path to dest ZooProcess dataset file")
    parser.add_argument("mapping_file", help="Filename for class name mapping.")
    parser.add_argument("--columns", default="*", help="Column for class name mapping.")

    args = parser.parse_args()
    
    print("Arguments:")
    for arg, val in vars(args).items():
        print(" {}: {}".format(arg, val))
    print()
    
    columns = args.columns.split(",")

    mapping = {}
    with open(args.mapping_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=',')
        
        selected_fieldnames = [name for name in reader.fieldnames
                               if any(fnmatch(name, pat)
                               for pat in columns) and name != "unique_name"]
                               
        print("Selected columns: " + ", ".join(selected_fieldnames))
        
        for row in reader:
            mapping[row["unique_name"]] = {name: row[name] for name in selected_fieldnames}

    with open(args.source, "r", encoding="utf-8-sig") as f_in, \
        open(args.dest, 'w', encoding='utf-8') as f_out:
         
        reader = csv.DictReader(f_in, delimiter=',')
        
        fieldnames = reader.fieldnames + ["name_" + name for name in selected_fieldnames]
        writer = csv.DictWriter(f_out, fieldnames, delimiter=',')
        
        writer.writeheader()

        for i, row in enumerate(reader):
            unique_name = row["unique_name"]
            
            for name in selected_fieldnames:
                row["name_" + name] = mapping.get(unique_name, {}).get(name, "")
                
            writer.writerow(row)
            
            if i % 100000 == 0 and i > 0:
                print("Processed {:,d} samples.".format(i))

    print("Done.")

if __name__ == "__main__":
    sys.exit(main())
