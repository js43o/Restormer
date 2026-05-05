from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (
    paired_paths_from_folder,
    paired_DP_paths_from_folder,
    paired_paths_from_lmdb,
    paired_paths_from_meta_info_file,
)
from basicsr.data.transforms import (
    augment,
    paired_random_crop,
    paired_random_crop_DP,
    random_augmentation,
)
from basicsr.data import degradations as degradations
from basicsr.utils import (
    FileClient,
    imfrombytes,
    img2tensor,
    padding,
    padding_DP,
    imfrombytesDP,
)

import random
import numpy as np
import torch
import cv2
import os
from glob import glob
import math

from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class Dataset_PairedImage(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_PairedImage, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None

        self.gt_folder, self.lq_folder = opt["dataroot_gt"], opt["dataroot_lq"]
        if "filename_tmpl" in opt:
            self.filename_tmpl = opt["filename_tmpl"]
        else:
            self.filename_tmpl = "{}"

        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt["client_keys"] = ["lq", "gt"]
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ["lq", "gt"]
            )
        elif "meta_info_file" in self.opt and self.opt["meta_info_file"] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder],
                ["lq", "gt"],
                self.opt["meta_info_file"],
                self.filename_tmpl,
            )
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ["lq", "gt"], self.filename_tmpl
            )

        if self.opt["phase"] == "train":
            self.geometric_augs = opt["geometric_augs"]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        scale = self.opt["scale"]
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]["gt_path"]
        img_bytes = self.file_client.get(gt_path, "gt")
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]["lq_path"]
        img_bytes = self.file_client.get(lq_path, "lq")
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))

        # augmentation for training
        if self.opt["phase"] == "train":
            gt_size = self.opt["gt_size"]
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {"lq": img_lq, "gt": img_gt, "lq_path": lq_path, "gt_path": gt_path}

    def __len__(self):
        return len(self.paths)


class Dataset_GaussianDenoising(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Dataset_GaussianDenoising, self).__init__()
        self.opt = opt

        if self.opt["phase"] == "train":
            self.sigma_type = opt["sigma_type"]
            self.sigma_range = opt["sigma_range"]
            assert self.sigma_type in ["constant", "random", "choice"]
        else:
            self.sigma_test = opt["sigma_test"]
        self.in_ch = opt["in_ch"]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None

        self.gt_folder = opt["dataroot_gt"]

        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = [self.gt_folder]
            self.io_backend_opt["client_keys"] = ["gt"]
            self.paths = paths_from_lmdb(self.gt_folder)
        elif "meta_info_file" in self.opt:
            with open(self.opt["meta_info_file"], "r") as fin:
                self.paths = [
                    osp.join(self.gt_folder, line.split(" ")[0]) for line in fin
                ]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        if self.opt["phase"] == "train":
            self.geometric_augs = self.opt["geometric_augs"]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        scale = self.opt["scale"]
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]["gt_path"]
        img_bytes = self.file_client.get(gt_path, "gt")

        if self.in_ch == 3:
            try:
                img_gt = imfrombytes(img_bytes, float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        else:
            try:
                img_gt = imfrombytes(img_bytes, flag="grayscale", float32=True)
            except:
                raise Exception("gt path {} not working".format(gt_path))

            img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = img_gt.copy()

        # augmentation for training
        if self.opt["phase"] == "train":
            gt_size = self.opt["gt_size"]
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)

            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

            if self.sigma_type == "constant":
                sigma_value = self.sigma_range
            elif self.sigma_type == "random":
                sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
            elif self.sigma_type == "choice":
                sigma_value = random.choice(self.sigma_range)

            noise_level = torch.FloatTensor([sigma_value]) / 255.0
            # noise_level_map = torch.ones((1, img_lq.size(1), img_lq.size(2))).mul_(noise_level).float()
            noise = torch.randn(img_lq.size()).mul_(noise_level).float()
            img_lq.add_(noise)

        else:
            np.random.seed(seed=0)
            img_lq += np.random.normal(0, self.sigma_test / 255.0, img_lq.shape)
            # noise_level_map = torch.ones((1, img_lq.shape[0], img_lq.shape[1])).mul_(self.sigma_test/255.0).float()

            img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        return {"lq": img_lq, "gt": img_gt, "lq_path": gt_path, "gt_path": gt_path}

    def __len__(self):
        return len(self.paths)


class Dataset_DefocusDeblur_DualPixel_16bit(data.Dataset):
    def __init__(self, opt):
        super(Dataset_DefocusDeblur_DualPixel_16bit, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.mean = opt["mean"] if "mean" in opt else None
        self.std = opt["std"] if "std" in opt else None

        self.gt_folder, self.lqL_folder, self.lqR_folder = (
            opt["dataroot_gt"],
            opt["dataroot_lqL"],
            opt["dataroot_lqR"],
        )
        if "filename_tmpl" in opt:
            self.filename_tmpl = opt["filename_tmpl"]
        else:
            self.filename_tmpl = "{}"

        self.paths = paired_DP_paths_from_folder(
            [self.lqL_folder, self.lqR_folder, self.gt_folder],
            ["lqL", "lqR", "gt"],
            self.filename_tmpl,
        )

        if self.opt["phase"] == "train":
            self.geometric_augs = self.opt["geometric_augs"]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        scale = self.opt["scale"]
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]["gt_path"]
        img_bytes = self.file_client.get(gt_path, "gt")
        try:
            img_gt = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lqL_path = self.paths[index]["lqL_path"]
        img_bytes = self.file_client.get(lqL_path, "lqL")
        try:
            img_lqL = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqL path {} not working".format(lqL_path))

        lqR_path = self.paths[index]["lqR_path"]
        img_bytes = self.file_client.get(lqR_path, "lqR")
        try:
            img_lqR = imfrombytesDP(img_bytes, float32=True)
        except:
            raise Exception("lqR path {} not working".format(lqR_path))

        # augmentation for training
        if self.opt["phase"] == "train":
            gt_size = self.opt["gt_size"]
            # padding
            img_lqL, img_lqR, img_gt = padding_DP(img_lqL, img_lqR, img_gt, gt_size)

            # random crop
            img_lqL, img_lqR, img_gt = paired_random_crop_DP(
                img_lqL, img_lqR, img_gt, gt_size, scale, gt_path
            )

            # flip, rotation
            if self.geometric_augs:
                img_lqL, img_lqR, img_gt = random_augmentation(img_lqL, img_lqR, img_gt)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lqL, img_lqR, img_gt = img2tensor(
            [img_lqL, img_lqR, img_gt], bgr2rgb=True, float32=True
        )
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lqL, self.mean, self.std, inplace=True)
            normalize(img_lqR, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lqL, img_lqR], 0)

        return {"lq": img_lq, "gt": img_gt, "lq_path": lqL_path, "gt_path": gt_path}

    def __len__(self):
        return len(self.paths)


#######################################################


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
        self.root = opt["dataroot"]

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


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == "resize_and_crop":
        new_h = new_w = opt.load_size
    elif opt.preprocess == "scale_width_and_crop":
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {"crop_pos": (x, y), "flip": flip}


def get_transform(
    opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True
):
    transform_list = []
    if grayscale:
        #  transform_list.append(transforms.Grayscale(1))
        from util import util

        transform_list.append(util.RGBtoY)
    if "resize" in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif "scale_width" in opt.preprocess:
        transform_list.append(
            transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method))
        )

    if "crop" in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            if "crop_size" in params:
                transform_list.append(
                    transforms.Lambda(
                        lambda img: __crop(img, params["crop_pos"], params["crop_size"])
                    )
                )
            else:
                transform_list.append(
                    transforms.Lambda(
                        lambda img: __crop(img, params["crop_pos"], opt.crop_size)
                    )
                )

    if opt.preprocess == "none":
        transform_list.append(
            transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method))
        )

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params["flip"]:
            transform_list.append(
                transforms.Lambda(lambda img: __flip(img, params["flip"]))
            )

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
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, "has_printed"):
        print(
            "The image size needs to be a multiple of 4. "
            "The loaded image size was (%d, %d), so it was adjusted to "
            "(%d, %d). This adjustment will be done to all images "
            "whose sizes are not multiples of 4" % (ow, oh, w, h)
        )
        __print_size_warning.has_printed = True


class BlindFFHQDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.img_size = opt["gt_size"]
        self.shuffle = True if opt["isTrain"] else False

        self.img_dir = opt["dataroot"]
        self.img_names = self.get_img_names()

        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        # degradations
        self.blur_kernel_size = opt["blur_kernel_size"]
        self.kernel_list = opt["kernel_list"]
        self.kernel_prob = opt["kernel_prob"]
        self.blur_sigma = opt["blur_sigma"]
        self.downsample_range = opt["downsample_range"]
        self.noise_range = opt["noise_range"]
        self.jpeg_range = opt["jpeg_range"]

    def get_img_names(
        self,
    ):
        img_names = []
        for ext in ["png", "jpg", "jpeg"]:
            img_names.extend([x for x in glob(os.path.join(self.img_dir, "*." + ext))])
            img_names.extend(
                [x for x in glob(os.path.join(self.img_dir, "**/*." + ext))]
            )

        img_names.sort()

        if self.shuffle:
            random.shuffle(img_names)

        print("# The number of images:", len(img_names))

        return img_names

    def __getitem__(self, index):
        # load gt image
        img_path = os.path.join(self.img_dir, self.img_names[index])
        hr_img = cv2.imread(img_path)
        hr_img = cv2.resize(
            hr_img, dsize=(512, 512), interpolation=cv2.INTER_LINEAR
        )  # resize for degradation
        hr_img = hr_img.astype(np.float32) / 255.0

        # ------------------------ generate lq image ------------------------ #
        # blur
        assert (
            self.blur_kernel_size[0] < self.blur_kernel_size[1]
        ), "Wrong blur kernel size range"
        cur_kernel_size = (
            random.randint(self.blur_kernel_size[0], self.blur_kernel_size[1]) * 2 + 1
        )
        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            cur_kernel_size,
            self.blur_sigma,
            self.blur_sigma,
            [-math.pi, math.pi],
            noise_range=None,
        )
        lr_img = cv2.filter2D(hr_img, -1, kernel)

        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        lr_img = cv2.resize(
            lr_img,
            (int(self.img_size // scale), int(self.img_size // scale)),
            interpolation=cv2.INTER_LINEAR,
        )

        # noise
        if self.noise_range is not None:
            lr_img = degradations.random_add_gaussian_noise(lr_img, self.noise_range)

        # jpeg compression
        if self.jpeg_range is not None:
            lr_img = degradations.random_add_jpg_compression(lr_img, self.jpeg_range)

        # resize to original size
        lr_img = cv2.resize(
            lr_img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )
        hr_img = cv2.resize(
            hr_img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )

        # BGR to RGB, HWC to CHW, numpy to tensor
        hr_img, lr_img = img2tensor([hr_img, lr_img], bgr2rgb=True, float32=True)

        # round and clip
        lr_img = torch.clamp((lr_img * 255.0).round(), 0, 255) / 255.0

        # normalize
        # normalize(hr_img, self.mean, self.std, inplace=True)
        # normalize(lr_img, self.mean, self.std, inplace=True)

        return {"gt": hr_img, "lq": lr_img, "gt_path": img_path, "lq_path": img_path}

    def __len__(self):
        return len(self.img_names)
