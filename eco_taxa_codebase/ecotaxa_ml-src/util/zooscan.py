"""
Load ZooProcess data.

ZooProcess data is text file with comma separated columns and a header of fieldnames.

See also features.py that loads SparseConvNet features.

@author: mschroeder
"""
import numpy as np
import os.path
import collections
from fnmatch import fnmatch

__all__ = ["load_datafile"]

# Fields of certain names are represented in a certain dtype
DTYPES_DEFAULT = {"unique_name": "<U256", "split": np.uint,
                  "objdate": "timedelta64[D]", "objtime":"<U8",
                  "name_*": "<U256", "objid": "<U32", "projid": "<U32", "imgid": "<U32", "id": "<U32"}

# Values to represent missing data
MISSING_VALUES = [(np.integer, 0), (np.floating, np.nan)]

class MatchingDict(collections.Mapping):
    """
    Dict-like class that retrieves entries that match a requested key.
    
    E.g. the contained key "name_*" matches "name_foo", "name_bar", ...
    """
    def __init__(self, dict_):
        self.dict = dict_
        
    def __getitem__(self, key):
        try:
            return self.dict[key]
        except KeyError:
            for k in self.dict:
                if fnmatch(key, k):
                    return self.dict[k]
            raise
            
    def __iter__(self):
        return iter(self.dict)
    
    def __len__(self):
        return len(self.dict)

def _get_missing_value(type_):
    for t, v in MISSING_VALUES:
        if np.issubdtype(type_, t):
            return v
    raise NotImplementedError("No missing value for " + repr(type_))

def _convert(type_, value):
    try:
        return type_(value)
    except ValueError:
        return _get_missing_value(type_)

def load_datafile(fn, default_dtype="float32", dtypes = None, replace_names = None, use_cache = True, verbose=True):
    """
    Load a ZooProcess data file.
    
    Because the loading of the data in the textfile is slow, the results can be
    cached.
    
    Parameters:
        fn: Filename
        default_dtype: Dtype for columns that don't match an entry in dtypes.
        dtypes: Dict of {pattern: dtype, ...} that specifies in which dtype a field should be represented.
        replace_names: Dict of {name: replacement, ...} for renaming fields.
        use_cache: Cache the result?
        verbose: Print info?
        
    Returns:
        A structured array with the fields in the file and the dtypes of the fields
        determined by dtypes or default_dtype.
    """
    cache_fn = fn + ".npy"
    
    if use_cache and os.path.isfile(cache_fn) and os.path.getmtime(cache_fn) > os.path.getmtime(fn):
        if verbose:
            print("Using cached version: {}.".format(cache_fn))
        return np.load(cache_fn)
    
    dtypes = dtypes or {}
    dtypes = MatchingDict({**DTYPES_DEFAULT, **dtypes})
    
    replace_names = replace_names or {}
    
    with open(fn, "r", encoding='utf-8-sig') as f:
        # Read header
        fieldnames = next(f).strip().split(",")
        fieldnames = [replace_names.get(n, n) for n in fieldnames]
        
        # Assemble dtype
        dtype = np.dtype([(n, dtypes.get(n, default_dtype)) for n in fieldnames])
        
        if verbose:
            print("Dtype is:", dtype)
        
        # Prepare types for conversion
        types = [np.dtype(dtypes.get(n, default_dtype)).type for n in fieldnames]
        
        n_lines = sum(1 for _ in f)
        
        # Create data array
        data = np.empty(n_lines, dtype)
        
        # Rewind file
        f.seek(0)
        
        # Skip header
        next(f)
        
        for i, line in enumerate(f):
            values = line.strip().split(",")
            data[i] = tuple(_convert(type_, value) for type_, value in zip(types, values))
    
    if use_cache:
        np.save(cache_fn, data)
    
    return data

def load_datafile_old(fn, default_dtype="float32", dtypes = None, replace_names = None):
    """
    DEPRECATED.
    """
    dtypes = dtypes or {}
    dtypes = {**DTYPES_DEFAULT, **dtypes}
    
    replace_names = replace_names or {}
    
    with open(fn, "r", encoding='utf-8-sig') as f:
        fieldnames = next(f).strip().split(",")
        
        fieldnames = [replace_names.get(n, n) for n in fieldnames]
        
        dtype = np.dtype([(n, dtypes.get(n, default_dtype)) for n in fieldnames])
        
        return np.genfromtxt((line.encode() for line in f), delimiter=",", dtype=dtype)
    
if __name__ == "__main__":
    # TODO: Test
    pass
