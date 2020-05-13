from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import types
import matplotlib.pyplot as plt
from utils import one_hot_encode
from plot import torch_to_img
from utils import compute_distribution_tensors


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
        self.lamb0 = 1
        self.lamb1 = 0
        self.lamb2 = 0
        self.model = model

    def __call__(
        self, output_l, label_l, outputs_p=None, labels_p=None, Yp=None, ref_distrib=None
    ):
        pred_l = F.log_softmax(output_l, dim=1)
        one_hot_func = partial(
            one_hot_encode, n_classes=pred_l.shape[1], type=pred_l.type()
        )
        label_l = one_hot_func(label_l)

        if outputs_p is None or (
            self.lamb1 == 0 and self.lamb2 == 0
        ):  # Avoid useless calculation
            return self.lamb0 * self.CE(pred_l, label_l)
        else:
            preds_p = [F.softmax(output_p, dim=1) for output_p in outputs_p]
            pred_distrib = compute_distribution_tensors(preds_p)
            preds_p = [torch.log(p + 1e-10) for p in preds_p]
            labels_p = [one_hot_func(label_p) for label_p in labels_p]

            Yp = [one_hot_func(yp) for yp in Yp]
        return (
            self.lamb0 * self.CE(pred_l, label_l)
            + self.lamb1 * self.Jp(preds_p, labels_p, Yp)
            + self.lamb2 * self.Jc(ref_distrib, pred_distrib)
        )

    def use_partial(self):
        """
        Switch to the 'second stage' of learning, using partial for gradient descent
        """
        self.lamb0 = 1
        self.lamb1 = 1.0
        self.lamb2 = 0.1  # TODO : add to config

    def switch_to_ascent(self):
        self.lamb0 = 0
        self.lamb1 = 0
        self.lamb2 = -self.lamb2

    def CE(self, output, label):
        """
        Basic Cross entropy loss on the fully labeled data

        Args:
            output (Tensor of size [Batch, NbClass, H, W]): Log of the softmax
                probability
            label (Tensor of size [Batch, NbClass, H, W]): Tensor indicating the
                class of a given pixel
        """
        height = output.shape[3]
        width = output.shape[2]
        loss = -torch.einsum("bcwh,bcwh->b", output, label)
        loss /= height * width
        return loss.mean()

    def Jp(self, outputs, labels_per_class, labels_predicted):
        """
        Basic Cross entropy loss on the partially labeled data + crossentropy
        on predicted label for the other classes -> Jp is our current estimation
        of the CE for the partially supervised dataset

        Args:
            outputs (List of Tensor of size [Batch, NbClass, H, W]): Log of the
                softmax probabilities.
            labels_per_class (List of Tensor of size [Batch, NbClass, H, W]):
                Tensors indicating if a given pixel correspond to the class.
            labels_predicted (List of Tensor of size [Batch, NbClass, H, W]):
                Tensors indicating the predicted class of a pixel.
        """
        height = outputs[0].shape[3]
        width = outputs[0].shape[2]
        loss = 0
        for out, label, label_p in zip(outputs, labels_per_class, labels_predicted):
            # CE on supervised
            l = torch.einsum("bcwh,bcwh->b", out, label)

            # CE on unsupervised, we extract the unsupervised layer from the predictions
            class_indicator = torch.sum(label, dim=(2, 3))
            # Background is a lack of information for partially supervised dataset
            class_indicator[:, 0] = 0
            norm = torch.max(class_indicator, dim=1, keepdim=True).values
            class_indicator = class_indicator / norm
            class_indicator = class_indicator.view(*label.shape[:2], 1, 1)
            class_indicator = class_indicator.expand_as(label_p)
            label_p = label_p * (1 - class_indicator) + label * class_indicator

            l += torch.einsum(
                "bcwh,bcwh->b", out, label_p
            )  # TODO this can be simplified in a single CE
            l /= height * width
            loss += l
        return -loss.mean()

    def Jc(self, q, p):
        # TODO this
        f = (q * self.model.nu - (1 - q) * self.model.mu) * p + q * torch.log(
            -self.model.nu
        )
        b = (1 - q) * (self.model.mu + torch.log(-self.model.mu))
        kl = torch.sum(f) + torch.sum(b)
        assert kl >= -100  # TODO Inspect this, it should not be < 0
        return kl


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
    assert true.shape[0] == logits.shape[0] and true.shape[1:] == logits.shape[2:]
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
    assert dice <= 1
    return dice  # TODO: Shouldn't be more than 1


def dice_loss(output, label, eps=1e-10):
    return 1 - dice(output, label, eps)
