# -*- coding: utf-8 -*-
"""
     implementation of PRNet DataLoader.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F

import cv2
from glob import glob
import random
import numbers
import numpy as np
from PIL import Image
from skimage import io

data_transform = {'train': transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}


class PRNetDataset(Dataset):
    """Pedestrian Attribute Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.dict = dict()
        self._max_idx()

    def get_img_path(self, img_id):
        id = self.dict.get(img_id)
        if id is not None:
            original = os.path.join(self.root_dir, str(id), 'original.jpg')
            #uv_map_path = None
            for file in os.listdir(os.path.join(self.root_dir,str(id))):
                if os.path.splitext(file)[1] == ".npy":
                    uv_map_path = os.path.join(self.root_dir,str(id),file)
                    break

            return original, uv_map_path
        else:
            print("aaaaaa",img_id)
            original = os.path.join(self.root_dir, str(0), 'original.jpg')
            for file in os.listdir(os.path.join(self.root_dir, str(0))):
                if os.path.splitext(file)[1] == ".npy":
                    uv_map_path = os.path.join(self.root_dir, str(0), file)
                    break
            return original,uv_map_path

    def _max_idx(self):
        _tmp_lst = map(lambda x: int(x), os.listdir(self.root_dir))
        _sorted_lst = sorted(_tmp_lst)
        for idx, item in enumerate(_sorted_lst):
            self.dict[idx] = item

    def __len__(self):
        return len(os.listdir(self.root_dir))-1

    def __getitem__(self, idx):
        original, uv_map = self.get_img_path(idx)

        # print(original)
        # print(uv_map)
        origin = cv2.imread(original)
        uv_map = np.load(uv_map)

        sample = {'uv_map': uv_map, 'origin': origin}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        uv_map, origin = sample['uv_map'], sample['origin']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        uv_map = uv_map.transpose((2, 0, 1))
        origin = origin.transpose((2, 0, 1))

        uv_map = uv_map.astype("float32") / 255.
        uv_map = np.clip(uv_map, 0, 1)
        origin = origin.astype("float32") / 255.
        
        return {'uv_map': torch.from_numpy(uv_map), 'origin': torch.from_numpy(origin)}


class ToNormalize(object):
    """Normalized process on origin Tensors."""

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        uv_map, origin = sample['uv_map'], sample['origin']
        origin = F.normalize(origin, self.mean, self.std, self.inplace)
        return {'uv_map': uv_map, 'origin': origin}
