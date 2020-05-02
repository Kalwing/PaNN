import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import types


def non_parametric_loss(loss, model):
    def r_loss(output, label):
        l = loss(output, label)
        return l

    return r_loss


def l2_loss(loss, model):
    def r_loss(output, label):
        w = 0
        for p in model.net_parameters:
            try:
                w += torch.norm(p.weight)
            except AttributeError:
                continue
        assert w > 0.6, w
        penalty = model.v * w
        l = loss(output, label)
        return l + penalty[0]

    return r_loss


class ZhouLoss:
    def __init__(self, model):
        self.lamb1 = 0
        self.lamb2 = 0

    def __call__(self, output, label):
        return (
            self.JL(output)
            + self.lamb1 * self.Jp(output, self.Yp)
            + self.lamb2 * self.Jc(output)
        )


def dice(logits, true, eps=1e-10):
    """Computes the Sørensen–Dice.

    Source: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    Args:
        true: a tensor of shape [B, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice: the Sørensen–Dice coeff.
    """
    true = true.unsqueeze(1)
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice = (2.0 * intersection / (cardinality + eps)).mean()
    return dice


def dice_loss(output, label, eps=1e-10):
    return 1 - dice(output, label, eps)
