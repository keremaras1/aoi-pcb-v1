import re
import numpy as np


def normalize_values(dataset):
    normalized_data = dataset / 255.0
    return normalized_data


def sort_alphanumeric(file_dir):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_dir, key=alphanum_key)


def rescale_values(dataset):
    rescaled_data = np.uint8(dataset * 255.0)
    return rescaled_data
