import os
import re
import numpy as np


def normalize_values(dataset):
    normalized_data = dataset / 255.0
    return normalized_data


def alphanum_key(file_name):
    """Convert file name to a tuple for sorting"""
    return tuple(int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', file_name))


# Sort file names in the dataset directory for accurate label generation
def sort_alphanumeric(file_dir):
    return sorted(os.listdir(file_dir), key=alphanum_key)


def rescale_values(dataset):
    rescaled_data = np.uint8(dataset * 255.0)
    return rescaled_data
