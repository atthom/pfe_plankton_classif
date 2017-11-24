#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Join arrays based on key columns.

@author: mschroeder
"""

import numpy as np

def _match_sorted_keys(key_left, key_right):
    """
    Match key_left to key_right. (Helper for join_* functions.)
    
    Parameters:
        key_left, key_right: 1d numpy arrays of shapes (N,) and (M,)
        
    Returns:
        A tuple of index arrays of the same shape where corresponding elements
        contain the indices in key_left and key_right for a matching value.
    """
    sorted_left = np.argsort(key_left)
    key_left = key_left[sorted_left]
    
    # Position of key_right in key_left
    idx1 = np.searchsorted(key_left, key_right, 'left') 
    idx2 = np.searchsorted(key_left, key_right, 'right')
    
    # Exact match if idx1[i] != idx2[i]
    exact_match = idx1 != idx2
    
    # left[i] -> right[j]: i = idx1[j] if exact_match[j]
    
    idx_left = idx1[exact_match]
    idx_right = np.where(exact_match)[0]
    
    assert len(idx_left) == len(idx_right)
    
    return sorted_left[idx_left], idx_right

def join_2d_arrays(left_keys, right_keys, left_values, right_values, return_indices = False):
    """
    Join two simple arrays (left_values, right_values) based on their keys (left_keys, right_keys).
    
    Parameters:
        left_keys, right_keys: 1d numpy arrays of shapes (N,) and (M,)
        left_values, right_values: 2d numpy arrays of shapes (N,*) and (M,*)
        return_indices: Return the indices of matching rows in the input in
        a addition to the merged array.
        
    Returns:
        Return an array with the columns of left_values, right_values.
        If return_indices, also return the corresponding indices
        (that enable the merging of meta data for example).
    """
    assert len(left_values.shape) == 2
    assert len(right_values.shape) == 2
    
    idx_left, idx_right = _match_sorted_keys(left_keys, right_keys)
    
    if not np.all(left_keys[idx_left] == right_keys[idx_right]):
        for i_left, i_right in zip(idx_left, idx_right):
            print(i_left, i_right, left_keys[i_left], right_keys[i_right])
        raise AssertionError("Keys do not match.")
    
    len_idx = len(idx_left)
    
    new_shape = (len_idx, left_values.shape[1] + right_values.shape[1])
    new_dtype = np.find_common_type([left_values.dtype, right_values.dtype], [])
    
    out = np.empty(new_shape, dtype=new_dtype)
    out[:,:left_values.shape[1]] = left_values[idx_left]
    out[:,left_values.shape[1]:] = right_values[idx_right]
    
    if return_indices:
        return out, idx_left, idx_right
    
    return out

def join_struct_flat(left, right_keys, right_values, on, right_name = "X"):
    idx_left, idx_right = _match_sorted_keys(left[on], right_keys)
    
    len_idx = len(idx_left)
    
    # assert np.all(left[on_left][idx_left] == right[on_right][idx_right])
    
    # print("{} matching rows.".format(len_idx))
    
    dtype_new = {}
    for descr in left.dtype.descr:
        dtype_new[descr[0]] = descr[1:]
        
    dtype_new[right_name] = (right_values.dtype.str, right_values.shape[1:])
        
    dtype_new = [(k, *v) for k, v in dtype_new.items()]
    dtype_new = np.dtype(dtype_new)
    
    out = np.empty(len_idx, dtype=dtype_new)
    
    for field in left.dtype.fields:
        out[field] = left[field][idx_left]
        
    out[right_name] = right_values[idx_right]
        
    return out

def join_recarrays(left, right, on=None, on_left=None, on_right=None):
    if on is not None:
        on_left = on_right = on
        
    idx_left, idx_right = _match_sorted_keys(left[on_left], right[on_right])
    
    len_idx = len(idx_left)
    
    # assert np.all(left[on_left][idx_left] == right[on_right][idx_right])
    
    # print("{} matching rows.".format(len_idx))
    
    dtype_new = {}
    for descr in left.dtype.descr:
        dtype_new[descr[0]] = descr[1:]
    for descr in right.dtype.descr:
        dtype_new[descr[0]] = descr[1:]
        
    dtype_new = [(k, *v) for k, v in dtype_new.items()]
    dtype_new = np.dtype(dtype_new)
    
    out = np.empty(len_idx, dtype=dtype_new)
    
    for field in left.dtype.fields:
        out[field] = left[field][idx_left]
        
    for field in right.dtype.fields:
        out[field] = right[field][idx_right]
        
    return out

if __name__ == "__main__":
#    from features import load_features2 as load_features
#    from zooscan import load_datafile
#    
#    zooprocess_data_fn = "/datapc/ob-ssd/mschroeder/Ecotaxa/Datasets/flowcam_data_train.csv"
#    scn_train_fn = "/datapc/ob/mschroeder/Ecotaxa/Results/SCN_flowcam_group1_0/_train.features"
#    
#    x = load_datafile(zooprocess_data_fn)
#    y = load_features(scn_train_fn)
#    
#    joined = join_struct_flat(x, y["objid"], y["X"], on="objid")

    a_keys = np.arange(2000)
    np.random.shuffle(a_keys)
    a_keys = a_keys[:1000]
    a_values = np.random.rand(1000, 20)
    
    shuffler = np.random.permutation(1000)
    b_keys = a_keys[shuffler]
    b_values = a_values[shuffler]
    
    joined = join_2d_arrays(a_keys, b_keys, a_values, b_values)
    
    joined_a = joined[:,:20]
    joined_b = joined[:,20:]
    assert np.all(joined_a == joined_b)