import torch
import numpy as np
import matplotlib.pyplot as plt

def non_parametric_loss(loss, model):
    def r_loss(output, label):
        l = loss(output, label)

        return  l

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
        return  l + penalty[0]

    return r_loss

class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs, target):
        log_p = (probs + 1e-10).log()
        mask = target.type(torch.float32)

        loss = - torch.einsum("bcwh,bcwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss