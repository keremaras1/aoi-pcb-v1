import csv
import numpy as np
from PIL import Image
from pathlib import Path
from aoi_pcb.data.utils import sort_alphanumeric, normalize_values


class DataEncoder:
    def __init__(self, config):
        self.size = config.encoder.train_data_splice
        self.normalize_data = config.encoder.normalize_data
        self.normalize_labels = config.encoder.normalize_labels
        self.dataset = []
        self.labels = []
        self.ref_coords = []
        self.ref_center = []

    def __call__(self, images_dir, labels_dir, *args, **kwargs):
        sorted_images = sort_alphanumeric(images_dir)

        self.dataset = self.image_to_numpy(Path(images_dir), sorted_images)
        self.labels, self.ref_coords, self.ref_center = self.coords_to_numpy(labels_dir, self.dataset.shape[1])

        if self.normalize_data:
            assert self.dataset.max() <= 1.0 and self.dataset.min() >= 0.0

        if self.normalize_labels:
            assert self.labels.max() <= 1.0 and self.labels.min() >= 0.0
            assert self.ref_coords.max() <= 1.0 and self.ref_coords.min() >= 0.0
            assert self.ref_center.max() <= 1.0 and self.ref_center.min() >= 0.0

        return self.dataset, self.labels, self.ref_coords, self.ref_center

    def image_to_numpy(self, images_dir, sorted_dir):
        dataset = []

        for file in sorted_dir:
            img = Image.open(images_dir / file)
            imgArray = np.array(img)
            dataset.append(imgArray)

        dataset = np.array(dataset)
        print("Input encoded...")

        if self.normalize_data:
            dataset = normalize_values(dataset)
            print("Input values normalized...")

        if self.size is not None:
            assert isinstance(self.size, int)
            normalized_dataset_split = dataset[:self.size]

            print("Data split to new size ", self.size)
            print("Shape: ", normalized_dataset_split.shape, " Type: ", type(normalized_dataset_split), " dtype: ",
                  normalized_dataset_split.dtype)

            return normalized_dataset_split

        print("Shape: ", dataset.shape, " Type: ", type(dataset), " dtype: ", dataset.dtype)

        return dataset

    def coords_to_numpy(self, csv_name, img_width=None):

        coords = []

        with open(csv_name, 'r') as file:
            csv_reader = csv.reader(file)
            csv_reader_list = list(csv_reader)

            ref_row = csv_reader_list[0]

            ref_points = list(
                int(x) for x in
                ref_row[0].replace("[", "").replace("]", "").replace("(", "").replace(")", "").split(','))

            ref_center = list(int(x) for x in ref_row[1].strip('()').split(','))

            for row in csv_reader_list[1:]:
                c_xy = []

                for tup in row:
                    c_xy.extend(int(x) for x in tup.strip('()').split(','))

                coords.append(c_xy)

        coords = np.asarray(coords)
        ref_points = np.asarray(ref_points)
        ref_center = np.asarray(ref_center)

        print("Labels generated from: ", csv_name)

        if self.normalize_labels and (img_width is not None):
            assert isinstance(img_width, int)

            coords = coords / img_width
            ref_points = ref_points / img_width
            ref_center = ref_center / img_width

            print("Labels normalized...")

        if self.size is not None:
            assert isinstance(self.size, int)

            coords_split = coords[:self.size]

            print("Labels split to new size: ", self.size)
            print("Shape: ", coords_split.shape, " Type: ", type(coords_split), " dtype: ", coords_split.dtype)

            return coords_split, ref_points, ref_center

        print("Shape: ", coords.shape, " Type: ", type(coords), " dtype: ", coords.dtype)

        return coords, ref_points, ref_center
