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

N_IMAGE = 5000
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
VAL_ID = N_IMAGE * TRAIN_SPLIT
TEST_ID = VAL_ID + N_IMAGE * VAL_SPLIT

DATA_DIR = DATA_FOLDER / "GEN"
os.makedirs(DATA_DIR, exist_ok=True)


def make_left_square(img):
    CLASS = 1
    center = (
        random.randint(10, IMG_WIDTH // 2 - 17),
        random.randint(10, IMG_WIDTH // 2 - 17),
    )
    radius = random.randint(7, 17)
    img = cv2.circle(img, center, radius, CLASS, -1)


# def draw_triangle(img, center, width):
#     """doesn't work yet"""
#     pts = np.array([
#          (center[0], center[1] + width),
#          (center[0] + width, center[1]),
#          (center[0] - width, center[1])
#     ], dtype=np.int32)
#     print(pts)
#     img = cv2.fillPoly(
#         img, pts, (255,255,255))

#     img = cv2.circle(img, center, radius, 255, -1)

# def draw_rectangle(img, center, width):
#     pts = np.array([
#          (center[0] - width, center[1] + width),
#          (center[0] + width, center[1] - width)
#     ], dtype=np.int32)
#     print(pts)
#     img = cv2.rectangle(img, pts[0], pts[1], 255)


def make_top_right_square(img):
    CLASS = 2
    center = np.array(
        (
            random.randint(IMG_WIDTH // 2 + 7, IMG_WIDTH - 7),
            random.randint(0 + 7, IMG_WIDTH // 2 - 7),
        )
    )
    radius = random.randint(7, 17)
    width = random.randint(min(4, int(radius * 0.24)), int(radius * 0.25))
    cv2.circle(img, tuple(center), radius, CLASS, width)


def make_bot_right_square(img):
    CLASS = 3
    center = np.array(
        (
            random.randint(IMG_WIDTH // 2 + 17, IMG_WIDTH - 17),
            random.randint(IMG_WIDTH // 2 + 17, IMG_WIDTH - 15),
        )
    )
    radius = random.randint(6, 17)
    shift = np.random.randint(min(4, radius * 0.75), radius * 0.9, (2,))
    cv2.circle(img, tuple(center), radius, CLASS, -1)
    cv2.circle(img, tuple(center + shift), radius, CLASS, -1)


for i in tqdm(range(N_IMAGE)):
    gt = np.zeros((IMG_WIDTH, IMG_WIDTH), np.uint16)
    make_left_square(gt)
    make_bot_right_square(gt)
    make_top_right_square(gt)

    img = ((gt > 0) * 1).astype(np.uint16)

    noise_amount = random.randint(0, 64)
    gaussian = (
        np.random.normal(0, noise_amount, (IMG_WIDTH, IMG_WIDTH))
    ).astype(np.float64)

    blur_size = tuple(
        [random.randrange(2, IMG_WIDTH // 3, 2) + 1 for x in range(2)]
    )

    # img = np.array(
    #     cv2.GaussianBlur(
    #         img * 255, tuple(blur_size), 0, borderType=cv2.BORDER_REFLECT
    #     )
    # )
    # img = img + gaussian
    img = img / img.max() * 255
    img = np.maximum(0, img).astype(np.uint8)
    if i < VAL_ID:
        path = DATA_DIR / "train"
    elif i < TEST_ID:
        path = DATA_DIR / "val"
    else:
        path = DATA_DIR / "test"
    os.makedirs(path / "gt", exist_ok=True)
    os.makedirs(path / "img", exist_ok=True)

    # assert np.max(gt) <= 1 and np.max(img) <= 1, (np.max(gt), np.max(img) )
    # assert len(np.unique(gt)) >= 2 and len(np.unique(img)) >= 2, "Image is binary"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(path / "gt" / f"{i}.png", gt)
    io.imsave(path / "img" / f"{i}.png", img)
