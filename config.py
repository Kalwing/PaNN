import networks
import torch.nn as nn
import torch.optim as optim
from functools import partial
from pathlib import Path
import losses

SEED = 1

DATA_FOLDER = Path("./data").absolute()
RESULTS_FOLDER = Path("./results")
SPLIT_FOLDERS = (Path("train"), Path("val"), Path("test"))
IMG_FOLDER = Path("img")
GT_FOLDER = Path("gt")

ARCHIVE_NAME = "mnist-original.zip"
DATA_NAME = "GEN"
TRAINING_NAME = f"{DATA_NAME}_CE"

DATASET_TYPE = "ImgPredDataset"
MODEL_NAME = f"{TRAINING_NAME}.pth"
TRAIN_CSV_NAME = f"{TRAINING_NAME}_results.csv"
TEST_CSV_NAME = f"{TRAINING_NAME}_testscore.csv"

NET = networks.SmallUNet
NET_PARAM = {"n_channels": 1, "n_classes": 4}
LOSS = partial(losses.non_parametric_loss, nn.CrossEntropyLoss())
OPTIM = partial(optim.SGD, lr=2e-2, momentum=0.8)
METRICS = {"Dice": losses.dice}

N_EPOCH = 10
BATCH_SIZE = 8
EARLY_STOP = 10
SAVE_INTERVAL = 5
