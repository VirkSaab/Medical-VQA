"""Source: https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
"""

import torch
import numpy as np

__all__ = [
    'mixup_data', 'mixup_criterion', 'mixup_data_vqa', 'mixup_criterion_vqa'
]


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data_vqa(v, q, a, alpha=1.0, use_cuda=True):
    '''Returns mixed images, pairs of questions, targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = v.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_v = lam * v + (1 - lam) * v[index, :]
    a_a, a_b = a, a[index]
    q_a, q_b = q, q[index]
    return mixed_v, a_a, a_b, q_a, q_b, lam


def mixup_criterion_vqa(criterion, pred_a, pred_b, a_a, a_b, lam):
    return lam * criterion(pred_a, a_a) + (1 - lam) * criterion(pred_b, a_b)
