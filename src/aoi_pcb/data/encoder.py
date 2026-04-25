"""Data encoding pipeline: loads images and CSV labels into numpy arrays."""

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from aoi_pcb.data.utils import normalize_values, sort_alphanumeric


class DataEncoder:
    """Encode a dataset of PCB images and keypoint labels into numpy arrays.

    Reads PNG images and a CSV label file, optionally normalizes pixel values
    and coordinates to [0, 1], and optionally truncates the dataset to a
    fixed size.

    Usage::

        encoder = DataEncoder(config)
        images, labels, ref_coords, ref_center = encoder(images_dir, labels_csv)
    """

    def __init__(self, config) -> None:
        """Initialize the encoder from a Config object.

        Args:
            config: Config instance; reads ``encoder.train_data_splice``,
                ``encoder.normalize_data``, and ``encoder.normalize_labels``.
        """
        self.size: int | None = config.encoder.train_data_splice
        self.normalize_data: bool = config.encoder.normalize_data
        self.normalize_labels: bool = config.encoder.normalize_labels

    def __call__(
        self,
        images_dir: str,
        labels_dir: str,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Run the encoding pipeline.

        Args:
            images_dir: Path to the directory containing PNG images.
            labels_dir: Path to the CSV labels file.

        Returns:
            Tuple of (images, labels, ref_coords, ref_center):
                - images: ``(N, H, W, 3)`` uint8 array in [0, 255] when
                  ``normalize_data`` is False; float64 array in [0, 1] when True.
                - labels: ``(N, 8)`` array of 4 keypoint (x, y) pairs.
                - ref_coords: ``(8,)`` reference corner coordinates.
                - ref_center: ``(2,)`` reference IC center coordinates.

        Raises:
            ValueError: If normalization is enabled but values are out of range.
        """
        sorted_images = sort_alphanumeric(images_dir)

        dataset = self.image_to_numpy(Path(images_dir), sorted_images)
        labels, ref_coords, ref_center = self.coords_to_numpy(labels_dir, dataset.shape[1])

        if self.normalize_data:
            if not (dataset.max() <= 1.0 and dataset.min() >= 0.0):
                raise ValueError(
                    "Image data is not normalized to [0, 1] after encoding."
                )  # pragma: no cover

        if self.normalize_labels:
            if not (labels.max() <= 1.0 and labels.min() >= 0.0):
                raise ValueError("Labels are not normalized to [0, 1].")
            if not (ref_coords.max() <= 1.0 and ref_coords.min() >= 0.0):
                raise ValueError("Reference coordinates are not normalized to [0, 1].")
            if not (ref_center.max() <= 1.0 and ref_center.min() >= 0.0):
                raise ValueError("Reference center is not normalized to [0, 1].")

        return dataset, labels, ref_coords, ref_center

    def image_to_numpy(
        self,
        images_dir: Path,
        sorted_dir: list[str],
    ) -> NDArray:
        """Load and stack images from a directory into a single numpy array.

        Args:
            images_dir: Path object pointing to the image directory.
            sorted_dir: Alphanumerically sorted list of filenames to load.

        Returns:
            ``(N, H, W, 3)`` numpy array. Normalised to [0, 1] if
            ``self.normalize_data`` is True, otherwise uint8 [0, 255].

        Raises:
            ValueError: If ``self.size`` is set but is not an integer.
        """
        dataset = []

        for file in sorted_dir:
            img = Image.open(images_dir / file)
            img_array = np.array(img)
            dataset.append(img_array)

        dataset = np.array(dataset)
        print("Input encoded...")

        if self.normalize_data:
            dataset = normalize_values(dataset)
            print("Input values normalized...")

        if self.size is not None:
            if not isinstance(self.size, int):
                raise ValueError(f"train_data_splice must be an integer, got {type(self.size)}.")
            dataset = dataset[: self.size]
            print("Data split to new size ", self.size)
            print("Shape: ", dataset.shape, " Type: ", type(dataset), " dtype: ", dataset.dtype)
            return dataset

        print("Shape: ", dataset.shape, " Type: ", type(dataset), " dtype: ", dataset.dtype)
        return dataset

    def coords_to_numpy(
        self,
        csv_name: str,
        img_width: int | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Parse a CSV label file into numpy arrays.

        The CSV is expected to have the reference keypoints and center on the
        first row, followed by one row of four (x, y) corner tuples per image.

        Args:
            csv_name: Path to the CSV file.
            img_width: Image width used to normalize coordinates when
                ``self.normalize_labels`` is True.

        Returns:
            Tuple of (coords, ref_points, ref_center):
                - coords: ``(N, 8)`` array of flattened corner coordinates.
                - ref_points: ``(8,)`` reference corner coordinates.
                - ref_center: ``(2,)`` reference IC center coordinates.

        Raises:
            ValueError: If ``self.size`` or ``img_width`` have unexpected types.
        """
        coords = []

        with open(csv_name, "r") as file:
            csv_reader = csv.reader(file)
            csv_reader_list = list(csv_reader)

            ref_row = csv_reader_list[0]

            ref_points = list(
                int(x)
                for x in ref_row[0]
                .replace("[", "")
                .replace("]", "")
                .replace("(", "")
                .replace(")", "")
                .split(",")
            )
            ref_center = list(int(x) for x in ref_row[1].strip("()").split(","))

            for row in csv_reader_list[1:]:
                c_xy = []
                for tup in row:
                    c_xy.extend(int(x) for x in tup.strip("()").split(","))
                coords.append(c_xy)

        coords = np.asarray(coords)
        ref_points = np.asarray(ref_points)
        ref_center = np.asarray(ref_center)

        print("Labels generated from: ", csv_name)

        if self.normalize_labels and img_width is not None:
            if not isinstance(img_width, int):
                raise ValueError(
                    f"img_width must be an integer, got {type(img_width)}."
                )  # pragma: no cover
            coords = coords / img_width
            ref_points = ref_points / img_width
            ref_center = ref_center / img_width
            print("Labels normalized...")

        if self.size is not None:
            if not isinstance(self.size, int):
                raise ValueError(
                    f"train_data_splice must be an integer, got {type(self.size)}."
                )  # pragma: no cover
            coords = coords[: self.size]
            print("Labels split to new size: ", self.size)
            print("Shape: ", coords.shape, " Type: ", type(coords), " dtype: ", coords.dtype)
            return coords, ref_points, ref_center

        print("Shape: ", coords.shape, " Type: ", type(coords), " dtype: ", coords.dtype)
        return coords, ref_points, ref_center
