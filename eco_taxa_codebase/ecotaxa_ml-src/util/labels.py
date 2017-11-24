#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LabelEncoder based on sklearn.preprocessing.LabelEncoder,
with the difference that it allows specifying a fixed list of classes and
does not allow fitting.

@author: mschroeder
"""

import numpy as np

class LabelEncoder(object):
    """Encode alphabetic labels into values between 0 and n_classes-1.

    Attributes:
		classes : array of shape (n_class,)
		    Holds the label for each class.
    """

    def __init__(self, classes=[]):
        self._classes = np.array(classes)
        self.order = np.argsort(self._classes)

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters:
		    y : array-like of shape [n_samples]
		        Target values.

        Returns:
	        y : array-like of shape [n_samples]
        """
        classes = np.unique(y)
        
        diff = np.setdiff1d(classes, self._classes)
        if len(diff) > 0:
            raise ValueError("y contains %d new labels: %s. Known labels are: %s." % (len(diff), str(diff), str(self._classes)))

        return np.searchsorted(self._classes, y, sorter=self.order)

    def inverse_transform(self, y):
        """Transform labels back to original encoding.
        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values.
        Returns
        -------
        y : numpy array of shape [n_samples]
        """

        y = np.asarray(y)
        
        diff = np.setdiff1d(y, np.arange(len(self._classes)))
        if len(diff) > 0:
            raise ValueError("y contains %d new labels: %s" % (len(diff), str(diff)))
        
        return self._classes[y]

    def encoded_labels(self):
        return np.arange(self._classes.shape[0])