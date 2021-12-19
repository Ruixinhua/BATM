import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy(output, target):
    return F.cross_entropy(output, target)


def categorical_loss(output, target, epsilon=1e-12):
    """
    Computes cross entropy between target (encoded as one-hot vectors) and output.
    Input: output (N, k) ndarray
           target (N, k) ndarray
    Returns: scalar
    """
    output, target = output.float(), target.float()
    output = torch.clamp(output, epsilon, 1. - epsilon)
    return -torch.sum(target * torch.log(output + 1e-9)) / output.shape[0]
