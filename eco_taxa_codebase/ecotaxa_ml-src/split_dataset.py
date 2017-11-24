#!/usr/bin/env python3
# encoding: utf-8
'''
Splits a ZooProcess data file into train and test set.

@author: Martin Schroeder <martin.schroeder@nerdluecht.de>
'''

import sys
from argparse import ArgumentParser
from collections import defaultdict
from itertools import count, cycle

def main():
    # Setup argument parser
    parser = ArgumentParser(description='''
    Splits a ZooProcess data file into train and test set.
    ''')
    parser.add_argument(
        "data",
        help="Path to data file")
    parser.add_argument(
        "train",
        help="Path to train data file")
    parser.add_argument(
        "test",
        help="Path to test data file")
    parser.add_argument(
        "--proportion",
        type=float,
        default=0.8,
        help="Proportion of training data")

    # Process arguments
    args = parser.parse_args()
    
    print("Arguments:")
    for arg, val in vars(args).items():
        print(" {}: {}".format(arg, val))
    print()

    prop_train = args.proportion
    prop_test = 1 - args.proportion
    factor = 1 / min(prop_train, prop_test)
    n_train = int(round(prop_train * factor))
    n_test = int(round(prop_test * factor))

    print(
        "Doing split using %d train samples on %d test sample..." %
        (n_train, n_test))

    # Initialize output distributor
    output_distributor = ['train'] * n_train + ['test'] * n_test

    # Initialize per output distributors.
    # This allows us to use next(per_class_output_distributors[cls])
    per_class_output_distributors = defaultdict(
        lambda: cycle(output_distributor))

    per_class_split_generators = defaultdict(lambda: count())

    with open(args.data, "r") as f_data, open(args.train, "w") as f_train, open(args.test, "w") as f_test:
        header = next(f_data)

        f_train.write(header.strip() + ",split\n")
        f_test.write(header)

        output_handles = {"train": f_train, "test": f_test}

        fieldnames = header.strip().split(",")

        try:
            label_index = fieldnames.index("unique_name")
        except ValueError:
            print("unique_name is not in header.", file=sys.stderr)
            raise

        for i, line in enumerate(f_data):
            values = line.strip().split(",")
            label = values[label_index]

            output_id = next(per_class_output_distributors[label])

            if output_id == "train":
                split_idx = next(per_class_split_generators[label])
                line = "%s,%s\n" % (line.strip(), split_idx)

            output_handles[output_id].write(line)

            if i > 0 and i % 100000 == 0:
                print("Processed {:,d} samples.".format(i))


if __name__ == "__main__":
    sys.exit(main())
