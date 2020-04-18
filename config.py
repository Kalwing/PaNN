import networks
import torch.nn as nn
import torch.optim as optim
from functools import partial
from pathlib import Path


DATA_FOLDER = Path('./data')
RESULTS_FOLDER = Path('./results')
SPLIT_FOLDERS = (Path('train'), Path('val'),  Path('test'))
IMG_FOLDER = Path('img')
GT_FOLDER = Path('gt')

ARCHIVE_NAME = 'mnist-original.zip'
DATA_NAME = 'MNIST'
TRAINING_NAME = F"{DATA_NAME}test"
MODEL_NAME = F'{TRAINING_NAME}.pth'
TRAIN_CSV_NAME = F'{TRAINING_NAME}_results.csv'
TEST_CSV_NAME = F'{TRAINING_NAME}_testscore.csv'

NET = networks.Net
NET_PARAM = {
    # "n_channels": 1,
    # "n_classes": 2
}
LOSS = nn.CrossEntropyLoss
OPTIM = partial(optim.SGD, lr=0.001, momentum=0.8)

N_EPOCH = 50
BATCH_SIZE = 4
EARLY_STOP = 5
SAVE_INTERVAL = 2