# -*- coding: utf-8 -*-
"""
Helper to turn a Numpy structured array (https://docs.scipy.org/doc/numpy-1.13.0/user/basics.rec.html)
of features into simple Numpy array with the features in columns.
"""

import warnings

def get_feature_array(arr, fields, dtype="float32"):
    """
    Turn a structured array of shape (N,) into a simple array
    of shape (N, len(fields)).
    
    Parameters
        arr: Structured array of features
        fields: Fields to select from arr
        dtype: Dtype of the result. All selected fields are coerced to this dtype.
    
    Returns
        Simple array view of shape (N, len(fields)) onto arr.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return arr[fields].view(dtype).reshape(arr.shape[0], -1)