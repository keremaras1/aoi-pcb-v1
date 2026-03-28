"""Utility functions for image normalization and alphanumeric file sorting."""

import os
import re
from typing import Final

import numpy as np
from numpy.typing import NDArray

MAX_PIXEL_VALUE: Final[float] = 255.0


def normalize_values(dataset: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Normalize image data from [0, 255] to [0.0, 1.0].

    Args:
        dataset: Array of uint8 image data.

    Returns:
        Float64 array with values in the range [0.0, 1.0].
    """
    return dataset / MAX_PIXEL_VALUE


def rescale_values(dataset: NDArray[np.float64]) -> NDArray[np.uint8]:
    """Rescale normalized image data from [0.0, 1.0] back to [0, 255].

    Args:
        dataset: Float64 array with values in the range [0.0, 1.0].

    Returns:
        uint8 array with values in the range [0, 255].
    """
    return np.uint8(dataset * MAX_PIXEL_VALUE)


def alphanum_key(file_name: str) -> tuple:
    """Convert a filename to a tuple key for alphanumeric sorting.

    Splits the filename on digit boundaries so that numeric parts are compared
    as integers rather than strings (e.g. ``pcb_9.png`` sorts before ``pcb_10.png``).

    Args:
        file_name: Filename string to convert.

    Returns:
        Tuple of alternating string and int segments.
    """
    return tuple(int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', file_name))


def sort_alphanumeric(file_dir: str) -> list[str]:
    """Return filenames in a directory sorted in alphanumeric order.

    Args:
        file_dir: Path to the directory to list.

    Returns:
        List of filenames sorted so that numeric segments are ordered numerically.
    """
    return sorted(os.listdir(file_dir), key=alphanum_key)
