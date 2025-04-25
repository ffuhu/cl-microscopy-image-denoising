import os
import torch
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
from natsort import natsorted


# import tifffile


def normalize_percentile(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32, verbose=False):
    """Percentile-based image normalization."""

    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    if verbose:
        print(f'[normalize_percentile]\tmin signal={mi.squeeze():.2f}\tmax signal={ma.squeeze():.2f}')
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = natsorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths

        # F210928: dirty hack to train with only 1 sample and allow higher batch sizes
        if self.opt.phase == 'train':
            self.n_samples = len(self.AB_paths)
            if 0 < self.opt.n_samples_train < self.opt.batch_size:
                print('\n\n\n\n(aligned_dataset.py@60 # dirty hack to train with n_samples < batch_size\n\n\n\n')
                self.AB_paths = self.AB_paths[50:50+self.opt.n_samples_train] * (len(self.AB_paths) // self.opt.n_samples_train)
            elif self.opt.batch_size <= self.opt.n_samples_train <= len(self.AB_paths):
                # F210824: to choose the number of training samples
                indices = list(range(len(self)))
                np.random.shuffle(indices)
                self.AB_paths = list(np.asarray(self.AB_paths)[indices[:self.opt.n_samples_train]]) * (len(self.AB_paths) // self.opt.n_samples_train)
            elif self.opt.n_samples_train != -1:
                print('Number of samples insuficient to sample.')
                raise NotImplementedError()
            self.AB_paths.extend(self.AB_paths[:self.n_samples - len(self.AB_paths)])

        if 'resize' in opt.preprocess or 'scale' in opt.preprocess:
            # crop_size should be smaller than the size of loaded image if resizing occurs
            assert (self.opt.load_size >= self.opt.crop_size)
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.img_dtype = opt.img_dtype

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        # F210927: added to be able to have batch_sizes > len(self.AB_paths)
        index = index % len(self.AB_paths)
        AB_path = self.AB_paths[index]
        # AB = Image.open(AB_path).convert('RGB')

        AB = Image.open(AB_path)
        # AB_np = np.array(AB_pil)
        # AB_np_uint16 = AB_np.astype('uint16')
        # AB = Image.fromarray(AB_np_uint16)
        # print('Image mode', AB.mode)

        # split AB image into A and B
        w, h = AB.size
        w2 = w // 2
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        if self.input_nc == 3:
            A = Image.fromarray(np.tile(np.asarray(A)[..., None], (1, 1, 3)).astype(self.img_dtype))
            B = Image.fromarray(np.tile(np.asarray(B)[..., None], (1, 1, 3)).astype(self.img_dtype))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A_transform = get_transform(self.opt, transform_params, grayscale=False, convert=False, target=False)
        B_transform = get_transform(self.opt, transform_params, grayscale=False, convert=False, target=True)

        A = A_transform(A)
        B = B_transform(B)

        # B = torch.tensor(np.asarray(B, np.float32))[None, ...] / 5.  # '/5.' to make labels {0, 0.2} instead of {0, 1}

        A = np.array(A, np.float32)[np.newaxis, ...] if self.input_nc == 1 else np.array(A, np.float32).transpose((2, 0, 1))
        A = torch.tensor(A)
        # A = (A - A.mean()) / A.std()

        B = np.array(B, np.float32)[np.newaxis, ...] if self.input_nc == 1 else np.array(B, np.float32).transpose((2, 0, 1))
        B = torch.tensor(B)
        # B = (B - B.mean()) / B.std()

        # Bc = torch.clip(B, -1, 1)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
