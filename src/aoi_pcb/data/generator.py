"""Synthetic PCB image dataset generation with randomised IC placement."""

import csv
import math
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from aoi_pcb.config_loader import Config

_IMAGE_SIZE: tuple[int, int] = (256, 256)
_ALPHA_FILL: int = 255
_DEFAULT_SCALE_FACTOR: float = 1.05


def add_tuple(at: tuple, bt: tuple) -> tuple:
    """Element-wise addition of two tuples.

    Args:
        at: First tuple.
        bt: Second tuple.

    Returns:
        Tuple where each element is the sum of the corresponding elements.
    """
    ret = ()
    for a, b in zip(at, bt):
        ret = ret + (a + b,)
    return ret


def add_alpha(img: np.ndarray) -> np.ndarray:
    """Append a fully opaque alpha channel to an RGB image.

    Args:
        img: HxWx3 uint8 image array.

    Returns:
        HxWx4 uint8 image array with alpha set to 255.
    """
    alpha = np.ones(img.shape[:-1] + (1,), dtype=np.uint8) * _ALPHA_FILL
    return np.concatenate((img, alpha), axis=-1)


def image_scale(
    img: np.ndarray,
    shape: tuple[int, int] | None = None,
    factor: float = _DEFAULT_SCALE_FACTOR,
) -> tuple[np.ndarray, float | None]:
    """Resize an image to an explicit shape or by a scale factor.

    Args:
        img: Input image array.
        shape: Target ``(width, height)`` in pixels. If None, ``factor`` is used.
        factor: Multiplicative scale factor applied when ``shape`` is None.

    Returns:
        Tuple of (resized image, rescale factor). The rescale factor is the
        ratio of the new width to the original width, or None when an explicit
        shape is provided.

    Raises:
        ValueError: If ``shape`` is provided but does not have exactly 2 elements.
    """
    h, w, _ = img.shape
    if shape is None:
        return cv2.resize(img, (int(w * factor), int(h * factor))), None
    else:
        if len(shape) != 2:
            raise ValueError(f"shape must have exactly 2 elements, got {len(shape)}.")
        return cv2.resize(img, shape), shape[1] / w


def pil_to_np(img: Image.Image) -> np.ndarray:
    """Convert a PIL RGBA image to a numpy array.

    Args:
        img: PIL Image in RGBA mode.

    Returns:
        ``(W, H, 4)`` uint8 numpy array.
    """
    return np.array(img.getdata(), dtype=np.uint8).reshape(img.size[0], img.size[1], 4)


def blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Alpha-composite image ``a`` over image ``b``.

    Args:
        a: Foreground RGBA image array (e.g. the IC).
        b: Background RGBA image array (e.g. the PCB).

    Returns:
        Composited RGB image as a contiguous uint8 array.
    """
    a = Image.fromarray(a)
    b = Image.fromarray(b)
    ret = Image.alpha_composite(b, a)
    ret = pil_to_np(ret)[:, :, :3]
    return np.ascontiguousarray(ret, dtype=np.uint8)


def org_corners(
    width: int,
    height: int,
    center: np.ndarray,
    angle: float,
) -> tuple[list[tuple], list[tuple]]:
    """Compute the original and rotated corner coordinates of the IC bounding box.

    Args:
        width: IC image width in pixels.
        height: IC image height in pixels.
        center: ``(2,)`` array representing the IC centre position on the PCB.
        angle: Rotation angle in radians.

    Returns:
        Tuple of (original corners, rotated corners), each a list of four
        ``(x, y)`` tuples in order: top-left, top-right, bottom-left, bottom-right.
    """
    tl = (0, 0)
    tr = add_tuple(tl, (width, 0))
    br = add_tuple(tl, (width, height))
    bl = add_tuple(tl, (0, height))
    theta = -angle
    c, s = np.cos(theta), np.sin(theta)
    r = np.array(((c, -s), (s, c)))
    original = [tl, tr, bl, br]
    return original, new_corners(original, r, width, height, center)


def new_corners(
    corners_og: list[tuple],
    R: np.ndarray,
    icw: int,
    ich: int,
    ic_center: np.ndarray,
) -> list[tuple]:
    """Apply a rotation matrix to IC corners and translate to the PCB coordinate frame.

    Args:
        corners_og: List of four ``(x, y)`` tuples at the IC origin.
        R: 2x2 rotation matrix.
        icw: IC image width in pixels.
        ich: IC image height in pixels.
        ic_center: ``(2,)`` array of the IC centre position on the PCB.

    Returns:
        List of four ``(x, y)`` tuples representing the rotated corners in PCB coordinates.
    """
    corner_mat = np.reshape(np.array(corners_og), (-1, 2))
    corner_mat = corner_mat - np.array([icw / 2, ich / 2])
    corner_mat = R.dot(corner_mat.T)
    corner_mat = corner_mat.T + ic_center
    corner_mat = corner_mat.astype("uint32")
    return [tuple(x) for x in corner_mat]


def get_layer(
    png: str,
    shape: tuple[int, int] | None = _IMAGE_SIZE,
    factor: float | None = None,
) -> tuple[int, int, float | None, np.ndarray]:
    """Load an image from disk and prepare it as an RGBA layer.

    Args:
        png: Path to the PNG file.
        shape: Target resize shape ``(width, height)``. Pass None to use ``factor`` instead.
        factor: Scale factor used when ``shape`` is None.

    Returns:
        Tuple of (height, width, rescale_factor, image array).
    """
    layer = cv2.imread(png)
    layer, rescale_factor = image_scale(layer, shape=shape, factor=factor)
    h, w, _ = layer.shape
    layer = add_alpha(layer)
    return h, w, rescale_factor, layer


def img_generator(
    alpha: float,
    delta_weight: float,
    backlayer_path: str,
    ic_path: str,
) -> tuple[np.ndarray, np.ndarray, list[tuple], float]:
    """Generate a single synthetic PCB image with a randomly placed IC.

    The IC is rotated by a uniformly sampled angle in ``[-alpha, alpha]`` degrees
    and displaced from centre by a random fraction of the available padding.

    Args:
        alpha: Maximum rotation angle in degrees.
        delta_weight: Fraction of available border space used for random displacement.
        backlayer_path: Path to the PCB background image.
        ic_path: Path to the IC component image.

    Returns:
        Tuple of (image, ic_center, corners, angle_degree):
            - image: ``(256, 256, 3)`` uint8 composited PCB image.
            - ic_center: ``(2,)`` float array of the IC centre position.
            - corners: List of four ``(x, y)`` rotated corner positions.
            - angle_degree: The sampled rotation angle in degrees.
    """
    angle_degree = np.random.uniform(-alpha, alpha)
    angle_radian = math.radians(angle_degree)

    back_height, back_width, rescale_factor, backlayer = get_layer(backlayer_path, shape=_IMAGE_SIZE)
    ic_height, ic_width, _, front_layer = get_layer(ic_path, shape=None, factor=rescale_factor)

    front_layer = ndimage.rotate(front_layer, angle_degree, reshape=True)
    front_height, front_width, _ = front_layer.shape

    width_delta = int((back_width - front_width) / 2)
    height_delta = int((back_height - front_height) / 2)
    cdx = random.randint(int(-width_delta * delta_weight), int(width_delta * delta_weight))
    cdy = random.randint(int(-height_delta * delta_weight), int(height_delta * delta_weight))

    ic_center = np.array([back_width / 2 + cdx, back_height / 2 + cdy])
    front_layer = cv2.copyMakeBorder(
        front_layer,
        height_delta + cdy, height_delta - cdy,
        width_delta + cdx, width_delta - cdx,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0, 0),
    )
    front_layer = cv2.resize(front_layer, (back_width, back_height))

    img = blend(front_layer, backlayer)
    _, corners = org_corners(ic_width, ic_height, ic_center, angle_radian)

    return img, ic_center, corners, angle_degree


def generate_dataset(
    dataset_size: int,
    rotation_angle: float,
    delta: float,
    data_dir: str,
    labels_dir: str,
    label_file: str,
    backlayer_path: str,
    ic_path: str,
    seed: int = 42,
) -> None:
    """Generate and save a synthetic PCB image dataset with keypoint labels.

    Creates ``dataset_size`` images by calling :func:`img_generator` with
    randomised rotation and displacement, writing each image to ``data_dir``
    and recording the four corner keypoints in a CSV file at ``labels_dir``.

    The first row of the CSV contains the reference keypoints (no rotation,
    no displacement) used to initialise the metric.

    Args:
        dataset_size: Number of images to generate.
        rotation_angle: Maximum IC rotation in degrees.
        delta: Maximum IC displacement as a fraction of available border space.
        data_dir: Directory where generated PNG images are saved.
        labels_dir: Directory where the CSV label file is written.
        label_file: Filename of the CSV label file.
        backlayer_path: Path to the PCB background image.
        ic_path: Path to the IC component image.
        seed: Random seed for reproducibility (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    label_dir = Path(labels_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / label_file

    _, center_org, ref_points, _ = img_generator(0, 0, backlayer_path, ic_path)

    with open(label_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([ref_points, tuple(center_org.astype(int))])
        print("Reference points written...")

        pbar = tqdm(desc='Generating images and labels...: ', total=dataset_size)
        for i in range(dataset_size):
            img, _, corners, _ = img_generator(rotation_angle, delta, backlayer_path, ic_path)
            img_path = data_dir / f"pcb_{i}.png"
            cv2.imwrite(str(img_path), img)
            writer.writerow(corners)
            pbar.update(1)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="libpng warning: iCCP: known incorrect sRGB profile")
    config = Config()

    backlayer = config.generator.image_sources.backlayer_path
    ic = config.generator.image_sources.ic_path

    if config.generator.gen_train:
        print("Generating training data...")
        train_args = config.get_init_kwargs("generator.train_data")
        generate_dataset(**train_args, backlayer_path=backlayer, ic_path=ic)

    if config.generator.gen_val:
        print("Generating validation data...")
        val_args = config.get_init_kwargs("generator.val_data")
        generate_dataset(**val_args, backlayer_path=backlayer, ic_path=ic)
