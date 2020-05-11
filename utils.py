import torch
from functools import partial


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


def compute_distribution_dl(dataloader):
    ref = next(iter(dataloader))[1]
    one_hot_func = partial(one_hot_encode, n_classes=torch.max(ref) + 1, type=ref.type())
    ref = one_hot_func(ref)

    gt_sum = torch.zeros(ref.shape[1:])
    n_pixel = ref.shape[-1] * ref.shape[-2]
    for _, gt in dataloader:
        one_hot = torch.sum(one_hot_func(gt), dim=0)
        gt_sum = gt_sum + one_hot / one_hot.shape[0]
    q = torch.einsum("lhw->l", gt_sum) / n_pixel

    return q / len(dataloader.dataset)


def compute_distribution_tensors(tensors):
    ref = tensors[0]
    p = torch.zeros(ref.shape[1]).type(ref.type())
    n_pixel = ref.shape[-1] * ref.shape[-2]
    for tensor in tensors:
        p += torch.einsum("blhw->l", tensor) / (n_pixel * ref.shape[0])

    return p / len(tensors)


def estimate_batch_mul_from_bs(batch_size, nb_partial_ds, full_ratio=3):
    batch_mul = batch_size // (nb_partial_ds * (1 + full_ratio))
    print(
        f"batch_mul={batch_mul}, the real Batch Size will be {nb_partial_ds * (1 + full_ratio) * batch_mul}"
    )
    return batch_mul
