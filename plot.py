import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path


def plot_results(csvs, save_path=None):
    if type(csvs) is not list:
        csvs = [csvs]
    for csv in csvs:
        df = pd.read_csv(csv)
        n = len(df.columns.values)
        fig, ax = plt.subplots(1, n - 1, figsize=(15, 5))
        for i in range(1, n):
            ax[i - 1].plot(df.iloc[:, 0], df.iloc[:, i])
            ax[i - 1].set_title(f"{df.columns.values[i]} per {df.columns.values[0]}")
        if save_path is None:
            plt.show()
        else:
            plt.savefig(Path(save_path) / f"{csv.stem}.png")
        purge_plt()


def plot_pred(dataset, model, n=2, device=None, save_path=None, prefix=""):
    inputs, labels = dataset[:n]

    fig, ax = plt.subplots(n, 3)
    ax[0, 0].set_title("Input")
    ax[0, 1].set_title("Ground Truth")
    ax[0, 2].set_title("Prediction")
    for i, input_ in enumerate(inputs):
        ax[i, 0].imshow(torch_to_img(inputs[i]))
        ax[i, 1].imshow(labels[i])

        to_predict = input_.unsqueeze(0)
        if device is not None:
            to_predict = to_predict.to(device)
        output = model(to_predict)[0].cpu().detach()
        pred = F.softmax(output, dim=0)
        # plt.imshow(pred[0])
        # plt.show()
        # plt.imshow(pred[1])
        # plt.show()
        output = torch_to_img(pred.detach().numpy())
        ax[i, 2].imshow(output)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(Path(save_path) / f"{prefix}{save_path.stem}_prediction.png")
    purge_plt()


def torch_to_img(tensor, normalize=True):
    if len(tensor.shape) > 2 and tensor.shape[0] > 1:
        # for c in range(tensor.shape[0]):
        #     tensor[c] = tensor[c] * c
        #     tensor[0] = torch.max(tensor[c], tensor[0])
        tensor = np.argmax(tensor, 0)
        # print("class detected", np.unique(tensor))
        if normalize:
            tensor = tensor / tensor.max()
            return tensor * 255.0
        return tensor
    else:
        return tensor.transpose(2, 0).squeeze(-1).T * 255.0


def purge_plt():
    plt.clf()
    plt.cla()
    plt.close()


def show_data(full_ds, partial_ds):
    fig, ax = plt.subplots(len(partial_ds) + 1, 2)

    ax[0, 0].imshow(torch_to_img(full_ds[0][0]))
    ax[0, 1].imshow(full_ds[0][1])
    ax[0, 0].set_ylabel("Fully supervised\n dataset", rotation=0, labelpad=35)
    ax[0, 0].set_title("Input")
    ax[0, 1].set_title("Ground Truth")
    for i, p in enumerate(partial_ds):
        ax[i + 1, 0].imshow(torch_to_img(p[0][0]))
        ax[i + 1, 1].imshow(p[0][1])
        ax[i + 1, 0].set_ylabel(
            f"Partially supervised\n dataset {p.name}", rotation=0, labelpad=35
        )

    plt.show()


# def show_data(full_ds, partial_ds):
#     fig, ax = plt.subplots(2, 4, figsize=(12, 5))

#     ax[0, 0].set_ylabel("Input", rotation=0, labelpad=35)
#     ax[1, 0].set_ylabel("Ground Truth", rotation=0, labelpad=35)
#     for i, p in enumerate(range(4)):
#         ax[0, i].imshow(torch_to_img(full_ds[i][0]))
#         ax[1, i].imshow(full_ds[i][1])

#     plt.show()
