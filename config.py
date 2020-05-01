import networks
import torch.nn as nn
import torch.optim as optim
from functools import partial
from pathlib import Path
import losses

DATA_FOLDER = Path("./data").absolute()
RESULTS_FOLDER = Path("./results")
SPLIT_FOLDERS = (Path("train"), Path("val"), Path("test"))
IMG_FOLDER = Path("img")
GT_FOLDER = Path("gt")

ARCHIVE_NAME = "mnist-original.zip"
DATA_NAME = "GEN"
TRAINING_NAME = f"{DATA_NAME}test"

DATASET_TYPE = "ImgPredDataset"
MODEL_NAME = f"{TRAINING_NAME}.pth"
TRAIN_CSV_NAME = f"{TRAINING_NAME}_results.csv"
TEST_CSV_NAME = f"{TRAINING_NAME}_testscore.csv"

NET = networks.SmallUNet
NET_PARAM = {"n_channels": 1, "n_classes": 1}
LOSS = partial(losses.non_parametric_loss, losses.DiceLoss())
OPTIM = partial(optim.SGD, lr=0.001, momentum=0.8)
METRICS = {"MSE": nn.MSELoss()}

N_EPOCH = 150
BATCH_SIZE = 8
EARLY_STOP = 5
SAVE_INTERVAL = 2
