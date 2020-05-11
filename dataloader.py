from functools import partial
import torch
import torchvision
import torchvision.transforms as transforms
import os
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import warnings
from skimage import io
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from config import (
    DATA_FOLDER,
    DATA_NAME,
    SPLIT_FOLDERS,
    GT_FOLDER,
    IMG_FOLDER,
    BATCH_SIZE,
    SEED,
)
from utils import one_hot_encode
from plot import torch_to_img


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
            (DATA_FOLDER / name / SPLIT_FOLDERS[split_id] / IMG_FOLDER).iterdir()
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
            + str([(label, form.format(f)) for label, f in zip(self.labels, self.dist)])
        )


class ImgPredDataset(Dataset):
    def __init__(
        self, name, transform=None, split_type="train", base_path=None, randomize=True
    ):
        """
        Args:
            transform (callable, optional): Transform to be applied on a sample.
                Defaults to None.
        """
        if base_path is None:
            base_path = DATA_FOLDER / DATA_NAME
        self.transform = transform
        self.name = name
        split_id = ["train", "val", "test"].index(split_type)
        self.gts = list(
            (base_path / name / SPLIT_FOLDERS[split_id] / GT_FOLDER).iterdir()
        )
        self.gts.sort(key=lambda p: p.stem)
        self.imgs = list(
            (base_path / name / SPLIT_FOLDERS[split_id] / IMG_FOLDER).iterdir()
        )
        self.imgs.sort(key=lambda p: p.stem)
        if randomize:
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
            # print(self.imgs[id_])
            img = io.imread(self.imgs[id_], as_gray=True).astype("float64")
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            img = img.transpose((2, 0, 1)) / 255.0

            gt = io.imread(str(self.gts[id_]))
            # if len(gt.shape) == 2:
            #     gt = np.expand_dims(gt, axis=-1)
            # gt = gt.transpose((2, 0, 1)) / 255.0
            assert np.max(gt) >= 1 and np.max(img) <= 1, (
                np.max(gt),
                np.max(img),
            )
            assert len(np.unique(gt)) >= 2, "gt is empty"
            assert img.shape[0] == 1, img.shape
            assert gt.shape[:2] == img.shape[1:], f"GT:{gt.shape}, IMG:{img.shape}"

            if self.transform:
                img = self.transform(img)
                gt = self.transform(gt)
            imgs.append(torch.from_numpy(img).type(torch.FloatTensor))
            gts.append(torch.from_numpy(gt).type(torch.LongTensor))
        if len(imgs) == 1:
            return imgs[0], gts[0]
        else:
            return torch.stack(imgs), torch.stack(gts)


class ZhouLoader:
    def __init__(self, full_ds, partial_ds, batch_mul, full_ratio=3, *args):
        """
        Create a dataloader returning for each batch:
            ({len(partial_ds)} * (1 + {full_ratio}) * {batch_mul}) images
        for each Batch

        Args:
            full_ds (Dataset): A fully supervised dataset
            partial_ds (List(Dataset)): A list containing the partially
                supervised datasets
            batch_mul (int): the number of pair taken from a partially
                supervised dataset
            full_ratio (int, optional): The number of fully supervised
                input/label pairs in each batch for each partially supervised
                pair. Defaults to 3.
        """
        self.full_loader = DataLoader(full_ds, batch_size=batch_mul * full_ratio)
        self.partial_loaders = [DataLoader(ds, batch_size=batch_mul) for ds in partial_ds]

    def __iter__(self):
        self.iter = 0
        self.partial_loaders_iter = [iter(p) for p in self.partial_loaders]
        self.full_loader_iter = iter(self.full_loader)
        self.ds_finished = [False for i in range(1 + len(self.partial_loaders))]
        return self

    def __next__(self):
        """
        Return a tuple of type (next element in Fully supervised dataset,
            (next elements in the partially supervised ones))
        Reset each iterator until all the iterator have been exhausted once
        """
        try:
            full_next = next(self.full_loader_iter)
        except StopIteration:
            self.ds_finished[0] = True
            self.full_loader_iter = iter(self.full_loader)
            full_next = next(self.full_loader_iter)

        partials_next = []
        for i, partial in enumerate(self.partial_loaders):
            try:
                partials_next.append(next(self.partial_loaders_iter[i]))
            except StopIteration:
                self.ds_finished[i + 1] = True
                self.partial_loaders_iter[i] = iter(self.partial_loaders[i])
                partials_next.append(next(self.partial_loaders_iter[i]))

        if sum(self.ds_finished) == len(self.ds_finished):  #  All dataset are finished
            raise StopIteration
        return (full_next, tuple(partials_next))

    def __len__(self):
        """
        The len of this loader is equal to the length of the longest dataset given
        """
        return max([len(dl) for dl in self.partial_loaders] + [len(self.full_loader)])


def estimate_batch_mul_from_bs(batch_size, nb_partial_ds, full_ratio=3):
    batch_mul = BATCH_SIZE // (nb_partial_ds * (1 + full_ratio))
    print(f"The real Batch Size will be {batch_mul}")
    return batch_mul


def get_dataloader(dataset):
    return DataLoader(dataset, batch_size=BATCH_SIZE)


def save_preds(train_loader, model, base_path, device=None):
    for i, dataset in enumerate(train_loader.partial_loaders):
        path = base_path / str(i + 1) / "train"
        os.makedirs(path, exist_ok=True)
        n = 0
        os.makedirs(path / "gt", exist_ok=True)
        os.makedirs(path / "img", exist_ok=True)
        for imgs, gts in dataset:
            for img, gt in zip(imgs, gts):
                to_predict = img.unsqueeze(0)
                if device is not None:
                    to_predict = to_predict.to(device)
                output = model(to_predict)[0].cpu().detach()
                assert torch.max(output) > 0
                pred = F.softmax(output, dim=0)
                output = torch_to_img(pred.detach().numpy(), normalize=False)
                img = torch_to_img(img.detach()).numpy().astype(np.uint16)

                assert 1 < np.max(img) <= 255, f"{np.max(output)} {np.max(img)}"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    io.imsave(path / "gt" / f"{n:05d}.png", output.astype(np.uint16))
                    io.imsave(path / "img" / f"{n:05d}.png", img)

                n += 1
