#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the distribution of object sizes (measured by bounding box diagonal).

Q: How is the diagonal length of images distributet?
A: Many small objects, long tail of larger objects.
    90% of objects are smaller than:
    flowcam: 253
    uvp5ccelter: 30
    zoocam: 377
    zooscan: 310

@author: mschroeder
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from util.zooscan import load_datafile
import os

plt.style.use('seaborn-colorblind')

dataset_root = "/datapc/ob-ssd/mschroeder/Ecotaxa/Datasets/"

for name in ["flowcam", "uvp5ccelter", "zoocam", "zooscan"]:
    print(name)
    suffixes = ["_data_with_dimensions.csv", "_data.csv"]
    fnames = [os.path.join(dataset_root, name + s) for s in suffixes]
    fname, *_ = [fname for fname in fnames if os.path.isfile(fname)]
    
    print(fname)
    dataset = load_datafile(fname)
    
    length = np.sqrt(dataset["width"]**2 + dataset["height"]**2)
    
    # Get 90% percentile
    p90 = np.percentile(length, 90)
    
    print("90% percentile:", p90)
    
    density = gaussian_kde(length)
    
    xs = np.linspace(0, 2000, 500)
    
    line, = plt.plot(xs, density(xs), label=name)
    color = line.get_color()
    
    plt.axvline(p90, 0, 1, c=color)
    
    plt.annotate("{:.0f}".format(p90), xy = (p90, density(p90)), xytext = (3,3), textcoords = "offset points")
    
plt.xlim(0)
plt.ylim(0)
plt.legend()
plt.savefig("size-distribution.pdf")