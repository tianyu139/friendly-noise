import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as data

import os
import sys
import torch
import pickle

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_target(args, transform_test):
    # get the target image from pickled file
    if args.backdoor:
        with open(os.path.join(args.poisons_path, "source.pickle"), "rb") as handle:
            target_images = pickle.load(handle)
            target_class = target_images[0][1]
            target_images = [transform_test(t[0]) for t in target_images]

        return target_images, target_class

    with open(os.path.join(args.poisons_path, "target.pickle"), "rb") as handle:
        target_img_tuple = pickle.load(handle)
        target_class = target_img_tuple[1]
        if len(target_img_tuple) == 4:
            patch = target_img_tuple[2] if torch.is_tensor(target_img_tuple[2]) else \
                torch.tensor(target_img_tuple[2])
            if patch.shape[0] != 3 or patch.shape[1] != args.patch_size or \
                    patch.shape[2] != args.patch_size:
                print(
                    f"Expected shape of the patch is [3, {args.patch_size}, {args.patch_size}] "
                    f"but is {patch.shape}. Exiting from poison_test.py."
                )
                sys.exit()

            startx, starty = target_img_tuple[3]
            target_img_pil = target_img_tuple[0]
            h, w = target_img_pil.size

            if starty + args.patch_size > h or startx + args.patch_size > w:
                print(
                    "Invalid startx or starty point for the patch. Exiting from poison_test.py."
                )
                sys.exit()

            target_img_tensor = transforms.ToTensor()(target_img_pil)
            target_img_tensor[:, starty : starty + args.patch_size,
                              startx : startx + args.patch_size] = patch
            target_img_pil = transforms.ToPILImage()(target_img_tensor)

        else:
            target_img_pil = target_img_tuple[0]

        target_img = transform_test(target_img_pil)
    return target_img, target_class


class PerturbedDataset(data.Dataset):
    def __init__(self):
        pass
