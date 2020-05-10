import torch


def one_hot_encode(label, n_classes, type):
    """
    Return the label one hot encoded

    Args:
        label (Tensor): Label of type B,H,W

    Returns:
        Tensor: [description]
    """
    label_1_hot = torch.eye(n_classes)[label.squeeze(1)]
    label_1_hot = label_1_hot.permute(0, 3, 1, 2).float()
    return label_1_hot.type(type)
