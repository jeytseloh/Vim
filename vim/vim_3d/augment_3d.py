import torch
from torchvision import transforms

from timm.data.transforms import RandomResizedCropAndInterpolation

import numpy as np
from torchvision import datasets

from utils.Data_Transform import Resize, RandomCrop, RandomFlip_LR, ToTensor, Normalize, Compose


def new_data_aug_generator_3d(args = None):
    img_size = args.input_size
    remove_random_resized_crop = args.src
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    primary_tfl = []
    scale=(0.08, 1.0)
    # interpolation='bicubic'
    interpolation='tricubic'
    if remove_random_resized_crop:
        primary_tfl = [
            Resize(img_size, interpolation=3),
            RandomCrop(img_size, padding=4,padding_mode='reflect'),
            RandomFlip_LR()
        ]
    else:
        primary_tfl = [
            RandomResizedCropAndInterpolation(
                img_size, scale=scale, interpolation=interpolation),
            RandomFlip_LR()
        ]

        
    # secondary_tfl = [transforms.RandomChoice([gray_scale(p=1.0),
    #                                           Solarization(p=1.0),
    #                                           GaussianBlur(p=1.0)])]
   
    # if args.color_jitter is not None and not args.color_jitter==0:
    #     secondary_tfl.append(transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter))
    final_tfl = [
            ToTensor(),
            Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
    return Compose(primary_tfl+final_tfl)