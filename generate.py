import PIL
import cv2
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from config import DATA_FOLDER
from tqdm import tqdm

IMG_WIDTH = 128

N_IMAGE = 1000
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
VAL_ID = N_IMAGE * TRAIN_SPLIT
TEST_ID = VAL_ID + N_IMAGE * VAL_SPLIT

DATA_DIR = DATA_FOLDER/'GEN'
os.makedirs(DATA_DIR, exist_ok=True)

def make_left_square(img):
    center = (random.randint(10, IMG_WIDTH//2 - 1), random.randint(10, IMG_WIDTH//2 - 1))
    radius = random.randint(7, 17)
    img = cv2.circle(img, center, radius, 255, -1)

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
    center = np.array((
              random.randint(IMG_WIDTH//2, IMG_WIDTH - 5),
              random.randint(0, IMG_WIDTH//2 - 5))
    )
    radius = random.randint(4, 17)
    width = random.randint(min(2, int(radius*0.24)), int(radius*0.25))
    cv2.circle(img, tuple(center), radius, 255, width)

def make_bot_right_square(img):
    center = np.array((
              random.randint(IMG_WIDTH//2, IMG_WIDTH - 7),
              random.randint(IMG_WIDTH//2, IMG_WIDTH - 5))
    )
    radius = random.randint(6, 17)
    shift = np.random.randint(min(4, radius*0.5), radius*0.85, (2,))
    cv2.circle(img, tuple(center), radius, 255, -1)
    cv2.circle(img, tuple(center+shift), radius, 255, -1)



for i in tqdm(range(N_IMAGE)):
    img = np.zeros((IMG_WIDTH,IMG_WIDTH), np.uint8)
    make_left_square(img)
    make_bot_right_square(img)
    make_top_right_square(img)

    gt = np.maximum(0, img)

    noise_amount = random.randint(0, 96)
    gaussian = np.maximum(
        1,
        np.random.normal(10, noise_amount, (IMG_WIDTH, IMG_WIDTH))
    ).astype(np.uint8)

    blur_size = tuple([
        random.randrange(0, IMG_WIDTH//3, 2) + 1 for x in range(2)
    ])

    img = cv2.GaussianBlur(img, tuple(blur_size),0)
    img += gaussian
    img = np.maximum(0, img)

    if i < VAL_ID:
        path = DATA_DIR/"train"
    elif i < TEST_ID:
        path = DATA_DIR/"val"
    else:
        path = DATA_DIR/"test"
    os.makedirs(path/'gt', exist_ok=True)
    os.makedirs(path/'img', exist_ok=True)

    # assert np.max(gt) <= 1 and np.max(img) <= 1, (np.max(gt), np.max(img) )
    # assert len(np.unique(gt)) >= 2 and len(np.unique(img)) >= 2, "Image is binary"
    io.imsave(path/'gt'/F"{i}.png", gt)
    io.imsave(path/'img'/F"{i}.png", img)


