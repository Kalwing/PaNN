import os
from scipy.io import loadmat
from sklearn.utils import shuffle
from pathlib import Path
import zipfile
from config import DATA_FOLDER, SPLIT_FOLDERS, IMG_FOLDER, GT_FOLDER, \
                   ARCHIVE_NAME, DATA_NAME
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np


def unzip_in_folder(path):
    folder_path = path.with_suffix('')
    if folder_path.exists():
        print(folder_path, "already unzipped")
    else:
        with zipfile.ZipFile(path, "r") as f:
            f.extractall(folder_path)
        print(path, "unzipped")
    return folder_path

def prepare_mnist(train_val_test_split = (0.7, 0.15, 0.15)):
    path = (DATA_FOLDER/ARCHIVE_NAME).resolve()
    print("Preparation of the dataset for training:")
    mnist_folder_path = unzip_in_folder(path)
    # Reading:
    data = loadmat(mnist_folder_path/'mnist-original.mat')
    X, y = data['data'], data['label']
    X = X.reshape(28,28, -1)
    X = np.moveaxis(X, -1, 0)
    y = np.moveaxis(y, -1, 0)
    X, y = shuffle(X, y)
    new_path = (DATA_FOLDER/DATA_NAME).resolve()

    # Saving:
    try:
        os.mkdir(new_path)
        for folder in SPLIT_FOLDERS:
            os.mkdir(new_path/folder)
            os.mkdir(new_path/folder/GT_FOLDER)
            os.mkdir(new_path/folder/IMG_FOLDER)
    except FileExistsError:
        return

    N = X.shape[0]
    split_id = 0
    last_split_size = 0
    for i in tqdm(range(N)):
        img = Image.fromarray(X[i])
        cur_split_size = train_val_test_split[split_id]*N
        if i >= last_split_size + cur_split_size:
            print(SPLIT_FOLDERS[split_id], "saved.")
            split_id += 1
            last_split_size = last_split_size + cur_split_size
        img.save(new_path/SPLIT_FOLDERS[split_id]/IMG_FOLDER/F"{i}.png")
        f = open(new_path/SPLIT_FOLDERS[split_id]/GT_FOLDER/F"{i}.txt",
            'w'
        )
        f.write(str(y[i][0]))
        f.close()



def prepare_segthor(easy=True):
    path = (DATA_FOLDER/ARCHIVE_NAME).resolve()
    print("Preparation of the dataset for training:")
    folder_path = unzip_in_folder(path)/'Thomas_Segthor'

    # Reading:
    train_path = folder_path/F'train_{"ancillary" if easy else "Primary"}'
    val_path = folder_path/'val'
    test_path = folder_path/'test'


    names_train = (train_path/'inst').iterdir()
    names_val = (val_path/'img').iterdir()
    names_test = (test_path/'img').iterdir()
    
    # Saving:
    try:
        os.mkdir(new_path)
        for folder in SPLIT_FOLDERS:
            os.mkdir(new_path/folder)
            os.mkdir(new_path/folder/GT_FOLDER)
            os.mkdir(new_path/folder/IMG_FOLDER)
    except FileExistsError:
        return

    N = X.shape[0]
    split_id = 0
    last_split_size = 0
    for i in tqdm(range(N)):
        img = Image.fromarray(X[i])
        cur_split_size = train_val_test_split[split_id]*N
        if i >= last_split_size + cur_split_size:
            print(SPLIT_FOLDERS[split_id], "saved.")
            split_id += 1
            last_split_size = last_split_size + cur_split_size
        img.save(new_path/SPLIT_FOLDERS[split_id]/IMG_FOLDER/F"{i}.png")
        f = open(new_path/SPLIT_FOLDERS[split_id]/GT_FOLDER/F"{i}.txt",
            'w'
        )
        f.write(str(y[i][0]))
        f.close()


if __name__ == "__main__":
    DATA_FUNC = {
        "MNIST", prepare_mnist,
        "SEGTHOR", prepare_segthor
    }
    DATA_FUNC[DATA_NAME]()