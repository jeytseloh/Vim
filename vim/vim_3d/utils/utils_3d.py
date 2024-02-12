from typing import Callable, List, Optional
from collections.abc import Sequence

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from monai.networks.layers import Conv
from monai.utils import ensure_tuple_rep

def adjust_window(image, centre, width):
    w_min = centre - width // 2
    w_max = centre + width // 2
    image = 255.0 * (image - w_min) / (w_max - w_min)
    image[image>255] = 255 # clip upper
    image[image<0] = 0 # clip lower
    return image

def zscore_normalization(image, mean=None, std=None):
    """
    Z-Score normalization.
    """
    if mean == None or std == None:
        raise Exception("Mean and std must be computed.")
    image = (image - mean) / std
    return image

def ct_normalization(image, p995=None, p005=None, mean=None, std=None):
    """
    Perform CT normalization. 
    Compute the mean, standard deviation, 0.5 and 99.5 percentile of the values. 
    Then clip to the percentiles, followed by subtraction of the mean and division with the standard deviation.
    (from nnUNet)
    """
    if p995 == None or p005 == None or mean == None or std == None:
        raise Exception("Mean and std must be computed.")
    image = torch.clamp(image, p005, p995)
    image -= mean
    image /= max(std, 1e-8) # avoid division by 0
    return image

def calculate_foreground_stats(segmentation: np.ndarray, images: np.ndarray):
    """
    images=image with multiple channels = shape (c, x, y(, z))
    """

    foreground_mask = segmentation[0] > 0

    for i in range(len(images)):
        foreground_pixels = images[i][foreground_mask]
        num_fg = len(foreground_pixels)

        percentile_995 = np.percentile(foreground_pixels, 99.5) if num_fg > 0 else np.nan
        percentile_005 = np.percentile(foreground_pixels, 0.5) if num_fg > 0 else np.nan
        mean = np.mean(foreground_pixels) if num_fg > 0 else np.nan
        std = np.std(foreground_pixels) if num_fg > 0 else np.nan

    return percentile_995, percentile_005, mean, std


class PatchEmbed3D(nn.Module):
    """
    Taken from monai.networks.blocks.patchembedding
    Patch embedding block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Unlike ViT patch embedding block: (1) input is padded to satisfy window size requirements (2) normalized if
    specified (3) position embedding is not used.

    Example::

        >>> from monai.networks.blocks import PatchEmbed
        >>> PatchEmbed(patch_size=2, in_chans=1, embed_dim=48, norm_layer=nn.LayerNorm, spatial_dims=3)
    """

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: Sequence[int] | int = 16,
        in_chans: int = 1,
        embed_dim: int = 48,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            patch_size: dimension of patch size.
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            norm_layer: normalization layer.
            spatial_dims: spatial dimension.
        """

        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        patch_size = ensure_tuple_rep(patch_size, spatial_dims) # eg: (16, 16, 16)
        self.patch_size = patch_size

        # added calculation of number of patches
        if img_size is not None:
            self.img_size = ensure_tuple_rep(img_size, spatial_dims)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        self.embed_dim = embed_dim
        self.proj = Conv[Conv.CONV, spatial_dims](
            in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            _, _, d, h, w = x_shape
            if w % self.patch_size[2] != 0:
                x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
            if h % self.patch_size[1] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
            if d % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))

        elif len(x_shape) == 4:
            _, _, h, w = x_shape
            if w % self.patch_size[1] != 0:
                x = F.pad(x, (0, self.patch_size[1] - w % self.patch_size[1]))
            if h % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[0] - h % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            x_shape = x.size()
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            if len(x_shape) == 5:
                d, wh, ww = x_shape[2], x_shape[3], x_shape[4]
                x = x.transpose(1, 2).view(-1, self.embed_dim, d, wh, ww)
            elif len(x_shape) == 4:
                wh, ww = x_shape[2], x_shape[3]
                x = x.transpose(1, 2).view(-1, self.embed_dim, wh, ww)
        return x

