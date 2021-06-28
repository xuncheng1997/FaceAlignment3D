# -*- coding: utf-8 -*-
"""
    implementation of Loss
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import *

import cv2
import numpy as np


def preprocess(mask):
    """
    :param mask: grayscale of mask.
    :return:
    """
    tmp = {}
    mask[mask > 0] = mask[mask > 0] / 16
    mask[mask == 15] = 16
    mask[mask == 7] = 8

    return mask


class WeightMaskLoss(nn.Module):
    """
        L2_Loss * Weight Mask
    """

    def __init__(self, mask_path):
        super(WeightMaskLoss, self).__init__()
        if os.path.exists(mask_path):
            self.mask = cv2.imread(mask_path, 0)
            self.mask = torch.from_numpy(preprocess(self.mask)).float().to("cuda")
        else:
            raise FileNotFoundError("Mask File Not Found! Please Check your Settings!")

    def forward(self, pred, gt):
        
        
        result = torch.mean(torch.pow((pred - gt), 2), dim=1)
        result = torch.mul(result, self.mask)       
        result = torch.mean(result,dim=0)
        
        result = torch.sum(result)

        return result


def INFO(*inputs):
    if len(inputs) == 1:
        print("[ PRNet ] {}".format(inputs))
    elif len(inputs) == 2:
        print("[ PRNet ] {0}: {1}".format(inputs[0], inputs[1]))


if __name__ == "__main__":
    INFO("Random Seed", 1)