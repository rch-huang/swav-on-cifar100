import random
import numpy as np
from PIL import ImageFilter

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MultiCropDataset(Dataset):
    def __init__(
        self,
        data_path,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        return_index=False,
        train=True,
        download=True,
    ):
        """
        CIFAR-100 版本的 MultiCropDataset：
        - 从 data_path 读取 CIFAR-100
        - 图像原始大小 32x32
        - 通过 RandomResizedCrop 直接采样为 224 / 96 等尺寸（相当于 upsample）
        - 使用与 ImageNet SwAV 相同的增广策略
        """
        super().__init__()

        # === 1) 底层数据集：CIFAR-100 ===
        self.dataset = datasets.CIFAR100(
            root='./',
            train=train,
            download=download,
            transform=None,   # 我们自己做 multi-crop transform
        )

        # optional: 限制数据集大小
        if size_dataset >= 0:
            self.indices = np.arange(len(self.dataset))[:size_dataset]
        else:
            self.indices = np.arange(len(self.dataset))

        self.return_index = return_index

        # === 2) 构造与 ImageNet 相同的增广策略 ===
        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
         
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            # 这一段完全沿用 SwAV ImageNet 的 policy，只是输入图像现在是 CIFAR-100
            trans.extend([
                transforms.Compose([
                    randomresizedcrop,
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Compose(color_transform),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ])
            ] * nmb_crops[i])

        self.trans = trans

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real_idx = self.indices[index]
        image, _ = self.dataset[real_idx]   # image 是 PIL.Image

        # 按多 crop transform 生成 N 个视图
        multi_crops = [trans(image) for trans in self.trans]

        if self.return_index:
            return real_idx, multi_crops
        return multi_crops


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(
        0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
    )
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort
