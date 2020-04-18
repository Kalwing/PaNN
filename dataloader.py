import torch
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader
from config import DATA_FOLDER, SPLIT_FOLDERS, GT_FOLDER, IMG_FOLDER, \
                   BATCH_SIZE

class ImgClassifDataset(Dataset):
    def __init__(self, name, transform=None, split_type="train"):
        """
        Args:
            transform (callable, optional): Transform to be applied on a sample.
                Defaults to None.
        """
        self.transform = transform
        split_id = ['train', 'val', 'test'].index(split_type)
        self.gts = list((DATA_FOLDER/name/SPLIT_FOLDERS[split_id]/
                    GT_FOLDER).iterdir())
        self.gts.sort(key=lambda p: p.stem)
        self.imgs = list((DATA_FOLDER/name/SPLIT_FOLDERS[split_id]/
                     IMG_FOLDER).iterdir())
        self.imgs.sort(key=lambda p: p.stem)

        assert self.imgs[0].stem == self.gts[0].stem, \
               F"Path misordered: {self.imgs[:3]}!={self.gts[:3]}"

        gts_value = [open(path, 'r').readline() for path in self.gts]
        self.labels, self.counts = np.unique(gts_value, return_counts=True)
        self.dist = self.counts / len(self.gts)

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, idx):
        img = io.imread(self.imgs[idx]).astype('float64')
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        img = img.transpose((2, 0, 1))

        f = open(self.gts[idx], 'r')
        gt = float(f.readline())


        if self.transform:
            img = self.transform(img)
        return (torch.from_numpy(img).type(torch.FloatTensor),
                torch.tensor(gt, dtype=torch.long))

    def print_dist(self, n_digits=5):
        form = "{:." + str(n_digits) + "}"
        print(F"dist:" + str([
            (label, form.format(f))
            for label, f in zip(self.labels, self.dist)
        ]))

r = ImgClassifDataset('MNIST', split_type="test")[0]

def get_dataloader(dataset):
    return DataLoader(dataset, batch_size=BATCH_SIZE)