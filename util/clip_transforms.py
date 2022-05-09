import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor

try:
    import accimage
except ImportError:
    accimage = None

import torchvision.transforms.functional as F
from torchvision import transforms

__all__ = ["ToClipTensor", "ClipRandomResizedCrop", "ClipColorJitter", "ClipRandomGrayscale", "ClipRandomHorizontalFlip", "ClipResize", "ClipCenterCrop", "ClipNormalize"
]

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

class ToClipTensor(object):
    """Convert a List of ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, clip):
        """
        Args:
            clip (List of PIL Image or numpy.ndarray): Clip to be converted to tensor.

        Returns:
            Tensor: Converted clip.
        """

        return [F.to_tensor(img) for img in clip]

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ClipRandomResizedCrop(transforms.RandomResizedCrop):
    def __call__(self, clip):
        """
        Args:
            clip (List of PIL Image or Tensor): Clip to be cropped and resized.

        Returns:
            List of PIL Image or Tensor: Randomly cropped and resized clip.
        """
        i, j, h, w = self.get_params(clip[0], self.scale, self.ratio)
        return [F.resized_crop(img, i, j, h, w, self.size, self.interpolation) for img in clip]


class ClipColorJitter(transforms.ColorJitter):
     def __call__(self, clip):
        """
        Args:
            clip (List of PIL Image or Tensor): Input clip.

        Returns:
            List of PIL Image or Tensor: Color jittered clip.
        """
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                clip = [F.adjust_brightness(img, brightness_factor) for img in clip]

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                clip = [F.adjust_contrast(img, contrast_factor) for img in clip]

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                clip = [F.adjust_saturation(img, saturation_factor) for img in clip]

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                clip = [F.adjust_hue(img, hue_factor) for img in clip]

        return clip


class ClipRandomGrayscale(transforms.RandomGrayscale):
    def __call__(self, clip):
        """
        Args:
            clip (List of PIL Image or Tensor): Clip to be converted to grayscale.

        Returns:
            List of PIL Image or Tensor: Randomly grayscaled clip.
        """
        num_output_channels = 1 if clip[0].mode == 'L' else 3
        if random.random() < self.p:
            return [F.to_grayscale(img, num_output_channels=num_output_channels) for img in clip]
        return clip


class ClipRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, clip):
        """
        Args:
            clip (List of PIL Image or Tensor): Clip to be flipped.

        Returns:
            List of PIL Image or Tensor: Randomly flipped clip.
        """
        if torch.rand(1) < self.p:
            return [F.hflip(img) for img in clip]
        return clip


class ClipNormalize(object):
    """Normalize a list of tensor images with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (List of Tensor): List of tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor list.
        """
        return [F.normalize(img, self.mean, self.std, self.inplace) for img in clip]

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ClipResize(transforms.Resize):
    """Resize the list of PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __call__(self, clip):
        """
        Args:
            clip (List of Tensor):: Clip to be scaled.

        Returns:
            List of PIL Image: Rescaled clip.
        """
        return [F.resize(img, self.size, self.interpolation) for img in clip]


class ClipCenterCrop(transforms.CenterCrop):
    """Crops the given list of PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, clip):
        """
        Args:
            clip (List of Tensor): Clip to be cropped.

        Returns:
            List of PIL Image: Cropped clip.
        """
        return [F.center_crop(img, self.size) for img in clip]


class ClipFirstCrop(transforms.CenterCrop):
    """Crops the given list of PIL Image at the 1/3.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, clip):
        """
        Args:
            clip (List of Tensor): Clip to be cropped.

        Returns:
            List of PIL Image: Cropped clip.
        """
        if isinstance(self.size, numbers.Number):
            self.size = (int(self.size), int(self.size))
        else:
            assert len(self.size) == 2, "Please provide only two dimensions (h, w) for size."
        image_width, image_height = clip[0].size
        crop_height, crop_width = self.size
        if crop_width > image_width or crop_height > image_height:
            msg = "Requested crop size {} is bigger than input size {}"
            raise ValueError(msg.format(self.size, (image_height, image_width)))

        return [img.crop((0, 0, crop_width, crop_height)) for img in clip]
        
        
class ClipThirdCrop(transforms.CenterCrop):
    """Crops the given list of PIL Image at the 3/3.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, clip):
        """
        Args:
            clip (List of Tensor): Clip to be cropped.

        Returns:
            List of PIL Image: Cropped clip.
        """
        if isinstance(self.size, numbers.Number):
            self.size = (int(self.size), int(self.size))
        else:
            assert len(self.size) == 2, "Please provide only two dimensions (h, w) for size."
        image_width, image_height = clip[0].size
        crop_height, crop_width = self.size
        if crop_width > image_width or crop_height > image_height:
            msg = "Requested crop size {} is bigger than input size {}"
            raise ValueError(msg.format(self.size, (image_height, image_width)))

        return [img.crop((image_width - crop_width, image_height - crop_height, image_width, image_height)) for img in clip]