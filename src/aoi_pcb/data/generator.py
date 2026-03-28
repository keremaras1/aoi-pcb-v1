import cv2
import csv
import random
import warnings
from aoi_pcb.config_loader import Config
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
import numpy as np
from PIL import Image
import math


def close_windows():
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def convert_tuple_to_int(tup):
    return tuple(map(lambda x: int(x), tup))


def add_tuple(at, bt):
    ret = ()
    for a, b in zip(at, bt):
        ret = ret + (a + b,)
    return ret


def add_alpha(img):
    alpha = np.ones(img.shape[:-1] + (1,), dtype=np.uint8) * 255
    img = np.concatenate((img, alpha), axis=-1)
    return img


def image_scale(img, shape=None, factor=1.05):
    h, w, _ = img.shape
    if shape is None:
        return cv2.resize(img, (int(w * factor), int(h * factor))), None
    else:
        assert len(shape) == 2
        return cv2.resize(img, shape), shape[1] / w


def cv_imshow(img):
    cv2.imshow('sample image', img)
    while True:
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            break
    close_windows()


def pil_to_np(img):
    ret = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[0], img.size[1], 4)
    return ret


def blend(a, b):
    a = Image.fromarray(a)
    b = Image.fromarray(b)
    ret = Image.alpha_composite(b, a)
    ret = pil_to_np(ret)[:, :, :3]
    ret = np.ascontiguousarray(ret, dtype=np.uint8)
    return ret


def show_img(img):
    img = Image.fromarray(img)
    img.show()


def org_corners(width, height, center, angle):
    tl = (0, 0)
    tr = add_tuple(tl, (width, 0))
    br = add_tuple(tl, (width, height))
    bl = add_tuple(tl, (0, height))
    theta = -angle
    c, s = np.cos(theta), np.sin(theta)
    r = np.array(((c, -s), (s, c)))
    return [tl, tr, bl, br], new_corners([tl, tr, bl, br], r, width, height, center)


def new_corners(corners_og, R, icw, ich, ic_center):
    corner_mat = np.reshape(np.array(corners_og), (-1, 2))
    corner_mat = corner_mat - np.array([icw / 2, ich / 2])
    corner_mat = R.dot(corner_mat.T)
    corner_mat = corner_mat.T + ic_center
    corner_mat = corner_mat.astype("uint32")
    corner_list = [tuple(x) for x in corner_mat]
    return corner_list


def get_layer(png, shape=(256, 256), factor=None):
    layer = cv2.imread(png)
    layer, rescale_factor = image_scale(layer, shape=shape, factor=factor)
    h, w, _ = layer.shape
    layer = add_alpha(layer)
    return h, w, rescale_factor, layer


def img_generator(alpha, delta_weight):
    angle_degree = np.random.uniform(-alpha, alpha)
    angle_radian = math.radians(angle_degree)
    back_height, back_width, rescale_factor, backlayer = get_layer('pcb_images/pcb_backlayer.png', shape=(256, 256))
    ic_height, ic_width, _, front_layer = get_layer('pcb_images/ic.png', shape=None, factor=rescale_factor)
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
        value=(0, 0, 0, 0)
    )
    front_layer = cv2.resize(front_layer, (back_width, back_height))
    img = blend(front_layer, backlayer)
    og_corners, corners = org_corners(ic_width, ic_height, ic_center, angle_radian)
    return img, ic_center, corners, angle_degree


def generate_dataset(dataset_size, rotation_angle, delta, data_dir, labels_dir, label_file):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    label_dir = Path(labels_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / label_file
    _, center_org, ref_points, _ = img_generator(0, 0)
    f = open(label_path, 'w')
    writer = csv.writer(f)
    writer.writerow([ref_points, tuple(center_org.astype(int))])
    print("Reference points written...")
    pbar = tqdm(desc='Generating images and labels...: ', total=dataset_size)
    for i in range(dataset_size):
        img, ic_center, corners, beta = img_generator(rotation_angle, delta)
        filename = "pcb_{idx}.png".format(idx=i)
        img_path = data_dir / filename
        cv2.imwrite(str(img_path), img)
        writer.writerow(corners)
        pbar.update(1)
    f.close()


if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="libpng warning: iCCP: known incorrect sRGB profile")
    config = Config()
    if config.generator.gen_train:
        print("Generating training data...")
        train_args = config.get_init_kwargs("generator.train_data")
        generate_dataset(**train_args)
    if config.generator.gen_val:
        print("Generating validation data...")
        val_args = config.get_init_kwargs("generator.val_data")
        generate_dataset(**val_args)
