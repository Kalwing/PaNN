import networks
import torch.nn as nn
import torch.optim as optim
from functools import partial
from pathlib import Path
import losses
from utils import estimate_batch_mul_from_bs

SEED = 1

DATA_FOLDER = Path("./data").absolute()
RESULTS_FOLDER = Path("./results")
SPLIT_FOLDERS = (Path("train"), Path("val"), Path("test"))
IMG_FOLDER = Path("img")
GT_FOLDER = Path("gt")

ARCHIVE_NAME = "mnist-original.zip"
DATA_NAME = "GEN"
PARTIAL_IDS = [1, 2, 3]
TRAINING_NAME = f"{DATA_NAME}_CEp"

DATASET_TYPE = "ImgPredDataset"
MODEL_NAME = f"{TRAINING_NAME}.pth"
TRAIN_CSV_NAME = f"{TRAINING_NAME}_results.csv"
TEST_CSV_NAME = f"{TRAINING_NAME}_testscore.csv"

NET = networks.SmallUNetZhou
NET_PARAM = {"n_channels": 1, "n_classes": 4}
LOSS = losses.ZhouLoss
OPTIM = partial(optim.Adam)
METRICS = {"Dice": losses.dice}

N_EPOCH_1 = 20  # N_epoch for the first stage
N_EPOCH_2_ASC = 20
N_EPOCH_2_DESC = 20
BATCH_SIZE = 32
BATCH_MUL = estimate_batch_mul_from_bs(BATCH_SIZE, len(PARTIAL_IDS))  # TODO calculate it
EARLY_STOP = 20
SAVE_INTERVAL = 5

SECOND_STAGE_ITER = 10
