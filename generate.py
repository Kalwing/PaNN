import PIL
import cv2
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from config import DATA_FOLDER
from tqdm import tqdm
import warnings

IMG_WIDTH = 128


datasets = {"Full": 30, "1": 40, "2": 40, "3": 40}
N_IMAGE = sum(datasets.values())
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
VAL_ID = datasets["Full"] * TRAIN_SPLIT
TEST_ID = VAL_ID + datasets["Full"] * VAL_SPLIT


DATA_DIR = DATA_FOLDER / "GEN"
os.makedirs(DATA_DIR, exist_ok=True)


def make_left_square(img):
    CLASS = 1
    tmp = img.copy()
    center = (
        random.randint(10, IMG_WIDTH // 2 - 17),
        random.randint(10, IMG_WIDTH // 2 - 17),
    )
    radius = random.randint(7, 17)
    return cv2.circle(tmp, center, radius, CLASS, -1)


def make_top_right_square(img):
    CLASS = 2
    tmp = img.copy()
    center = np.array(
        (
            random.randint(IMG_WIDTH // 2 + 7, IMG_WIDTH - 7),
            random.randint(0 + 7, IMG_WIDTH // 2 - 7),
        )
    )
    radius = random.randint(7, 17)
    width = random.randint(min(4, int(radius * 0.24)), int(radius * 0.25))
    return cv2.circle(tmp, tuple(center), radius, CLASS, width)


def make_bot_right_square(img):
    CLASS = 3
    tmp = img.copy()
    center = np.array(
        (
            random.randint(IMG_WIDTH // 2 + 17, IMG_WIDTH - 17),
            random.randint(IMG_WIDTH // 2 + 17, IMG_WIDTH - 15),
        )
    )
    radius = random.randint(6, 17)
    shift = np.random.randint(min(4, radius * 0.75), radius * 0.9, (2,))
    cv2.circle(tmp, tuple(center), radius, CLASS, -1)
    return cv2.circle(tmp, tuple(center + shift), radius, CLASS, -1)


def generate_img_from_gt(gt, easy=False):
    img = ((gt > 0) * 1).astype(np.int32) * 255.0

    if not easy:
        blur_size = tuple([random.randrange(2, IMG_WIDTH // 3, 2) + 1 for x in range(2)])
        img = np.array(
            cv2.GaussianBlur(img, tuple(blur_size), 0, borderType=cv2.BORDER_REFLECT)
        )

        noise_amount = random.randint(0, 180)
        gaussian = (np.random.normal(0, noise_amount, (IMG_WIDTH, IMG_WIDTH))).astype(
            np.float64
        )
        img = img + gaussian

        img = img / img.max() * 255
    img = np.maximum(0, img).astype(np.uint8)
    return img


class_func = [make_left_square, make_bot_right_square, make_top_right_square]


for dataset in datasets.keys():
    for i in tqdm(range(datasets[dataset])):
        gt = np.zeros((IMG_WIDTH, IMG_WIDTH), np.uint16)

        if dataset == "Full":
            gt_labeled = gt
            for gen in class_func:
                gt_labeled = np.maximum(gt_labeled, gen(gt))
            gt_full = gt_labeled
        else:
            id_dataset = int(dataset) - 1
            # Partially labeled ds contain only the label of 1 class
            gt_labeled = class_func[id_dataset](gt)
            # We add the other to serve as reference for the img construction
            gt_full = gt_labeled
            for id_gen, gen in enumerate(class_func):
                if id_gen != id_dataset:
                    gt_full = np.maximum(gt_full, gen(gt))

        img = generate_img_from_gt(gt_full)
        path = DATA_DIR / dataset
        if dataset == "Full":
            if i < VAL_ID:
                path /= "train"
            elif i < TEST_ID:
                path /= "val"
            else:
                path /= "test"
        else:
            path /= "train"
        os.makedirs(path / "gt", exist_ok=True)
        os.makedirs(path / "img", exist_ok=True)

        assert np.max(gt_labeled) >= 1, np.max(gt_labeled)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(path / "gt" / f"{i}.png", gt_labeled)
        io.imsave(path / "img" / f"{i}.png", img)
