# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
from torch import NoneType
import pickle
from torchvision import datasets, transforms
import torch
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import glob
from PIL import Image, ImageFile
from torch.utils import data

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None



def crop_center(big_image, new_size):
    old_size = max(big_image.size) // 256
    start = (old_size - new_size) // 2 * 256
    end = start + new_size * 256
    return big_image.crop((start, start, end, end))


import concurrent.futures



def gen_context_img_siguo(cur_info):
    try:
        len1 = len(cur_info.split(" ")[0])
        len2 = len(cur_info.split(" ")[-1])
        filename = cur_info[:len1]
        clsid = cur_info[-len2:]
        clsname = cur_info[len1 + 1:-len2 - 1]

    except:
        print(cur_info, flush=True)
        exit()

    big_image55 = Image.open(r"/media/dell/DATA/wyn/new_code/try_last_check_1123/" + clsname + "/" + filename)
    big_image33 = crop_center(big_image55, 3)
    small_image = crop_center(big_image55, 1)

    return small_image, big_image33, big_image55


import numpy as np
import cv2


class WYNDataset(data.Dataset):
    def __init__(self, root, train=True, val=True, transform=None, transform33=None, transform55=None, vis=False):

        self.context18 = None
        self.context23 = None
        self.vis = vis
        if train:
            with open(r"/media/dell/DATA/wyn/MEET_train_1125.txt") as f:
                train_infos = f.readlines()
            f.close()

            trn_files = []
            trn_targets = []
            trn_lines = []

            for item in train_infos:
                len1 = len(item.split(" ")[0])
                len2 = len(item.split(" ")[-1])
                fname = item[:len1]
                idx = item[-len2:]
                #clsname = item[len1 + 1:-len2 - 1]
                trn_files.append(fname)
                trn_targets.append(int(idx))
                trn_lines.append(item)

            self.lines = trn_lines
            self.files = trn_files
            self.targets = trn_targets
        elif val:
            with open(r"/media/dell/DATA/wyn/MEET_valid_1125.txt") as f:
                valid_infos = f.readlines()
            f.close()

            val_files = []
            val_targets = []
            val_lines = []

            for item in valid_infos:
                len1 = len(item.split(" ")[0])
                len2 = len(item.split(" ")[-1])
                fname = item[:len1]
                idx = item[-len2:]
                val_files.append(fname)
                val_targets.append(int(idx))
                val_lines.append(item)


            self.files = val_files
            self.targets = val_targets
            self.lines = val_lines
        else:
            with open(r"/media/dell/DATA/wyn/MEET_test_1125.txt") as f:
                test_infos = f.readlines()
            f.close()

            test_files = []
            test_targets = []
            test_lines = []

            for item in test_infos:
                len1 = len(item.split(" ")[0])
                len2 = len(item.split(" ")[-1])
                fname = item[:len1]
                idx = item[-len2:]
                #clsname = item[len1 + 1:-len2 - 1]
                test_files.append(fname)
                test_targets.append(int(idx))
                test_lines.append(item)

            self.files = test_files
            self.targets = test_targets
            self.lines = test_lines

        self.transform = transform
        self.transform33 = transform33
        self.transform55 = transform55

    def __len__(self):
        return len(self.files)

    def get_wyn(self, i):
        cur_info = self.lines[i]
        img, img33, img55 = gen_context_img_siguo(cur_info)

        return img, img33, img55

    def __getitem__(self, i):

        img, img33, img55 = self.get_wyn(i)


        img = self.transform(img)
        img33 = self.transform33(img33)
        img55 = self.transform55(img55)
        return img, img33, img55, self.targets[i]



def build_dataset(is_train, is_val, args, vis=False):
    transform = build_transform(is_train, args)
    transform33 = build_transform33(is_train, args)
    transform55 = build_transform55(is_train, args)
    data_path = r'F:\dataset_landuse'
    args.nb_classes = 80
    dataset = WYNDataset(data_path, train=is_train, val=is_val, transform=transform, transform33=transform33,
                         transform55=transform55, vis=vis)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    t = []
    size = 224
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.ToTensor())
    return transforms.Compose(t)


def build_transform55(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    t = []
    size = 224
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )

    t.append(transforms.ToTensor())
    return transforms.Compose(t)


def build_transform33(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # eval transform
    t = []
    size = 224
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.ToTensor())
    return transforms.Compose(t)
