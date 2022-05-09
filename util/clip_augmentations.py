# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
# https://github.com/ildoonet/pytorch-randaugment/master/RandAugment/augmentations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image


def ShearX(clip, v):  # [0, 0.3]
    assert 0 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return [img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0)) for img in clip]


def ShearY(clip, v):  # [0, 0.3]
    assert 0 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return [img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0)) for img in clip]


def TranslateXabs(clip, v):  # [0, 100]
    assert 0 <= v <= 100
    if random.random() > 0.5:
        v = -v
    return [img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)) for img in clip]


def TranslateX(clip, v):  # [0, 0.4464]
    assert 0 <= v <= 0.4464
    if random.random() > 0.5:
        v = -v
    v = v * clip[0].size[0]
    return [img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)) for img in clip]


def TranslateYabs(clip, v):  # [0, 100]
    assert 0 <= v <= 100
    if random.random() > 0.5:
        v = -v
    return [img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)) for img in clip]


def TranslateY(clip, v):  # [0, 0.4464]
    assert 0 <= v <= 0.4464
    if random.random() > 0.5:
        v = -v
    v = v * clip[0].size[1]
    return [img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)) for img in clip]


def Rotate(clip, v):  # [0, 30]
    assert 0 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return [img.rotate(v) for img in clip]


def AutoContrast(clip, _):
    return [PIL.ImageOps.autocontrast(img) for img in clip]


def Invert(clip, _):
    return [PIL.ImageOps.invert(img) for img in clip]


def Equalize(clip, _):
    return [PIL.ImageOps.equalize(img) for img in clip]


def Solarize(clip, v):  # [0, 256]
    assert 0 <= v <= 256
    return [PIL.ImageOps.solarize(img, 256 - v) for img in clip]


def SolarizeAdd(clip, v): # [0, 110]
    assert 0 <= v <= 110
    threshold = 128

    clip_out = []
    for img in clip:
        img_raw = np.array(img)
        img_add = img_raw.astype(np.int)
        img_add = img_add + v
        img_add = np.clip(img_add, 0, 255)
        img_add = img_add.astype(np.uint8)
        img_np = np.where(img_raw < threshold, img_add, img_raw)
        clip_out.append(Image.fromarray(img_np))
    return clip_out


def Posterize(clip, v):  # [0, 4]
    assert 0 <= v <= 4
    v = int(v)
    return [PIL.ImageOps.posterize(img, 4 - v) for img in clip]


def Contrast(clip, v):  # [0, 0.9]
    assert 0 <= v <= 0.9
    if random.random() > 0.5:
        v = -v
    return [PIL.ImageEnhance.Contrast(img).enhance(1 - v) for img in clip]


def Color(clip, v):  # [0, 0.9]
    assert 0 <= v <= 0.9
    if random.random() > 0.5:
        v = -v
    return [PIL.ImageEnhance.Color(img).enhance(1 - v) for img in clip]


def Brightness(clip, v):  # [0, 0.9]
    assert 0 <= v <= 0.9
    if random.random() > 0.5:
        v = -v
    return [PIL.ImageEnhance.Brightness(img).enhance(1 - v) for img in clip]


def Sharpness(clip, v):  # [0, 0.9]
    assert 0 <= v <= 0.9
    if random.random() > 0.5:
        v = -v
    return [PIL.ImageEnhance.Sharpness(img).enhance(1 - v) for img in clip]


def Cutout(clip, v):  # [0, 0.3571]
    assert 0.0 <= v <= 0.3571

    v = v * clip[0].size[0]
    return CutoutAbs(clip, v)


def CutoutAbs(clip, v):  # [0, 80]
    # assert 0 <= v <= 80
    if v < 0:
        return clip

    w, h = clip[0].size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)

    clip_out = []
    for img in clip:
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
        clip_out.append(img)
    return clip_out


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),

        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),

        (Color, 0, 0.9),
        (Contrast, 0, 0.9),
        (Brightness, 0, 0.9),
        (Sharpness, 0, 0.9),

        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (Cutout, 0, 0.3571),
        (TranslateX, 0., 0.4464),
        (TranslateY, 0., 0.4464),
    ]

    return l


def _is_pil_image(img):
    return isinstance(img, Image.Image)


class ClipRandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, clip):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            clip = op(clip, val)
        return clip