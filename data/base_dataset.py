"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


# def get_params(opt, size):
#     w, h = size
#     new_h = h
#     new_w = w
#     if opt.preprocess == 'resize_and_crop':
#         new_h = new_w = opt.load_size
#     elif opt.preprocess == 'scale_width_and_crop':
#         new_w = opt.load_size
#         new_h = opt.load_size * h // w
#
#     x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
#     y = random.randint(0, np.maximum(0, new_h - opt.crop_size))
#
#     flip = random.random() > 0.5
#
#     angle = random.choice(opt.rot_angles) if 'rotate' in opt.preprocess else 0
#
#     return {'crop_pos': (x, y), 'flip': flip, 'rotate': angle}

# def get_transform(opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
#     transform_list = []
#     if grayscale:
#         transform_list.append(transforms.Grayscale(1))
#     if 'resize' in opt.preprocess:
#         osize = [opt.load_size, opt.load_size]
#         transform_list.append(transforms.Resize(osize, method))
#     elif 'scale_width' in opt.preprocess:
#         transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))
#
#     if 'crop' in opt.preprocess:
#         if params is None:
#             transform_list.append(transforms.RandomCrop(opt.crop_size))
#         else:
#             transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
#
#     if opt.preprocess == 'none':
#         transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))
#
#     if not opt.no_flip:
#         if params is None:
#             transform_list.append(transforms.RandomHorizontalFlip())
#         elif params['flip']:
#             transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
#
#     if 'rotate' in opt.preprocess:
#         if params is None:
#             transform_list.append(transforms.RandomRotation(90))
#         elif 'rotate' in params:
#             transform_list.append(transforms.Lambda(lambda img: __rotate(img, params['rotate'])))
#
#     if convert:
#         transform_list += [transforms.ToTensor()]
#         if grayscale:
#             transform_list += [transforms.Normalize((0.5,), (0.5,))]
#         else:
#             transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     return transforms.Compose(transform_list)

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x, y = None, None
    if 'crop' in opt.preprocess:
        x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
        y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    hflip = random.random() < opt.h_flip_prob if 'hflip' in opt.preprocess else False
    vflip = random.random() < opt.v_flip_prob if 'vflip' in opt.preprocess else False

    rot_angle = random.choice(opt.rot_angles) if 'rotate' in opt.preprocess else 0

    zoom = np.random.uniform(opt.random_zoom[0], opt.random_zoom[1]) if 'randomzoom' in opt.preprocess else 1

    gauss_noise_std = 0
    if 'gaussnoise' in opt.preprocess and random.random() < opt.gauss_noise_prob:
        if len(opt.gauss_noise_std) == 1:
            gauss_noise_std = opt.gauss_noise_std
        else:
            gauss_noise_std = np.random.randint(opt.gauss_noise_std[0], opt.gauss_noise_std[1], 1)[0]

    return {'crop_pos': (x, y), 'hflip': hflip, 'vflip': vflip, 'rot_angle': rot_angle, 'zoom': zoom,
            'gauss_noise_std': gauss_noise_std}


def get_transform(opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True,
                  target=False):
    transform_list = []

    # if phase is test, don't modify the input images
    if opt.phase == 'test':
        return transforms.Compose(transform_list)

    # if phase is train, augment the images
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if params['hflip']:
        transform_list.append(transforms.RandomHorizontalFlip(p=1))

    if params['vflip']:
        transform_list.append(transforms.RandomVerticalFlip(p=1))

    if 0 < params['rot_angle'] < 360:
        transform_list.append(transforms.Lambda(lambda img: __rotate(img, params['rot_angle'])))

    if not target:
        if params['gauss_noise_std'] > 0:
            transform_list.append(transforms.Lambda(lambda img: __add_gaussian_noise(img, params['gauss_noise_std'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __rotate(img, angle):
    return transforms.functional.rotate(img, angle)


def __zoom(img, scale, target=False):
    ow, oh = img.size
    nw, nh = int(ow * scale), int(oh * scale)

    img_res = img.resize((nw, nh), Image.BILINEAR if target else Image.NONE)
    if scale < 1:
        nimg = np.zeros_like(img)
        x = random.randint(0, ow - nw)
        y = random.randint(0, oh - nh)
        nimg[x:x + nw, y:y + nh] = img_res
    else:
        x = random.randint(0, np.maximum(0, nw - ow))
        y = random.randint(0, np.maximum(0, nh - oh))
        nimg = np.asarray(img_res)[x:x + oh, y:y + oh]

    return Image.fromarray(nimg)


def __add_gaussian_noise(img, scale=1., loc=0.):
    old_dtype = img.mode
    img_ = np.asarray(img)
    random_noise = np.random.normal(loc=loc, scale=scale, size=img_.shape)
    img_ = img_ + random_noise
    # assert img.dtype in [np.uint8, np.float32, np.float64], f'Image dtype not valid: {img.dtype}'
    # a_min = 0  # if img.dtype == np.uint8 else 0.
    # a_max = 255  # if img.dtype == np.uint8 else 1.
    # img = np.clip(img, a_min=a_min, a_max=a_max)
    img_ = Image.fromarray(img_.astype(np.uint16 if '16' in old_dtype else np.uint8))
    return img_


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
