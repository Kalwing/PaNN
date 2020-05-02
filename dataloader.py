import torch
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from config import (
    DATA_FOLDER,
    SPLIT_FOLDERS,
    GT_FOLDER,
    IMG_FOLDER,
    BATCH_SIZE,
    SEED,
)


class ImgClassifDataset(Dataset):
    def __init__(self, name, transform=None, split_type="train"):
        """
        Args:
            transform (callable, optional): Transform to be applied on a sample.
                Defaults to None.
        """
        self.transform = transform
        split_id = ["train", "val", "test"].index(split_type)
        self.gts = list(
            (DATA_FOLDER / name / SPLIT_FOLDERS[split_id] / GT_FOLDER).iterdir()
        )
        self.gts.sort(key=lambda p: p.stem)
        self.imgs = list(
            (
                DATA_FOLDER / name / SPLIT_FOLDERS[split_id] / IMG_FOLDER
            ).iterdir()
        )
        self.imgs.sort(key=lambda p: p.stem)

        assert (
            self.imgs[0].stem == self.gts[0].stem
        ), f"Path misordered: {self.imgs[:3]}!={self.gts[:3]}"

        gts_value = [open(path, "r").readline() for path in self.gts]
        self.labels, self.counts = np.unique(gts_value, return_counts=True)
        self.dist = self.counts / len(self.gts)

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, idx):
        img = io.imread(self.imgs[idx]).astype("float64")
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        img = img.transpose((2, 0, 1))

        f = open(self.gts[idx], "r")
        gt = float(f.readline())

        if self.transform:
            img = self.transform(img)
        return (
            torch.from_numpy(img).type(torch.FloatTensor),
            torch.tensor(gt, dtype=torch.long),
        )

    def print_dist(self, n_digits=5):
        form = "{:." + str(n_digits) + "}"
        print(
            f"dist:"
            + str(
                [
                    (label, form.format(f))
                    for label, f in zip(self.labels, self.dist)
                ]
            )
        )


class ImgPredDataset(Dataset):
    def __init__(self, name, transform=None, split_type="train"):
        """
        Args:
            transform (callable, optional): Transform to be applied on a sample.
                Defaults to None.
        """
        self.transform = transform
        split_id = ["train", "val", "test"].index(split_type)
        self.gts = list(
            (DATA_FOLDER / name / SPLIT_FOLDERS[split_id] / GT_FOLDER).iterdir()
        )
        self.gts.sort(key=lambda p: p.stem)
        self.imgs = list(
            (
                DATA_FOLDER / name / SPLIT_FOLDERS[split_id] / IMG_FOLDER
            ).iterdir()
        )
        self.imgs.sort(key=lambda p: p.stem)

        self.gts, self.imgs = shuffle(self.gts, self.imgs, random_state=SEED)
        assert (
            self.imgs[0].stem == self.gts[0].stem
        ), f"Path misordered: {self.imgs[:3]}!={self.gts[:3]}"

    def __len__(self):
        return len(self.gts)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else -1
            step = idx.step if idx.step is not None else 1
            idx = range(start, stop, step)
        else:
            idx = [idx]

        imgs = []
        gts = []
        for id_ in idx:
            img = io.imread(self.imgs[id_]).astype("float64")
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            img = img.transpose((2, 0, 1)) / 255.0

            gt = io.imread(self.gts[id_]).astype("float64")
            # if len(gt.shape) == 2:
            #     gt = np.expand_dims(gt, axis=-1)
            # gt = gt.transpose((2, 0, 1)) / 255.0

            assert np.max(gt) >= 2 and np.max(img) <= 1, (
                np.max(gt),
                np.max(img),
            )
            assert len(np.unique(gt)) > 2, "gt is empty"

            # assert gt.shape == img.shape, self.gts[id_]

            if self.transform:
                img = self.transform(img)
                gt = self.transform(gt)
            imgs.append(torch.from_numpy(img).type(torch.FloatTensor))
            gts.append(torch.from_numpy(gt).type(torch.LongTensor))
        if len(imgs) == 1:
            return imgs[0], gts[0]
        else:
            return torch.stack(imgs), torch.stack(gts)


def get_dataloader(dataset):
    return DataLoader(dataset, batch_size=BATCH_SIZE)
