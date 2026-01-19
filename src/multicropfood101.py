import random
import numpy as np
from PIL import ImageFilter

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


# ------------------------------------------------------------
# TaskSubset (UNCHANGED)
# ------------------------------------------------------------
class TaskSubset(Dataset):
    def __init__(self, base_dataset, selected_indices):
        self.base_dataset = base_dataset
        self.selected_indices = selected_indices

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, i):
        return self.base_dataset[self.selected_indices[i]][0]


# ------------------------------------------------------------
# MultiCropDataset (Food-101)
# ------------------------------------------------------------
class MultiCropDataset(Dataset):
    def __init__(
        self,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        *,
        train=True,
        val=False,
        val_size=5000,
        CICL=True,
        return_index=False,
        seed=0,
    ):
        super().__init__()

        self.train = train
        self.val = val
        self.CICL = CICL
        self.return_index = return_index

        split = "train" if train else "validation"
        self.dataset = load_dataset(
            "ethz/food101",
             split=split,
            cache_dir="/Scratch/repository/rh539/hf")

        # --------------------------------------------------
        # Keep only classes [0..99], drop class 100
        # --------------------------------------------------
        labels = self.dataset["label"]   # fast, vectorized
        self.indices = np.where(np.asarray(labels) < 100)[0]

        self.indices = np.asarray(self.indices, dtype=np.int64)

        if val:
            rng = np.random.RandomState(seed)
            rng.shuffle(self.indices)

            labels = np.asarray(self.dataset["label"])[self.indices]
            num_per_class = val_size // 100

            selected_pos = []
            for c in range(100):
                pos = np.where(labels == c)[0]
                selected_pos.append(pos[:num_per_class])

            selected_pos = np.concatenate(selected_pos)
            self.indices = self.indices[selected_pos]

        # --------------------------------------------------
        # Multi-crop transforms (same structure as Tiny-IN)
        # --------------------------------------------------
        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        self.biggest_crop = size_crops[0]
        self.trans = []

        if train and not val:
            for i in range(len(size_crops)):
                self.trans.extend(
                    [
                        transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomResizedCrop(
                                size_crops[i],
                                scale=(min_scale_crops[i], max_scale_crops[i]),
                            ),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.Compose(color_transform),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std),
                        ])
                    ] * nmb_crops[i]
                )
        else:
            self.trans.append(
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(self.biggest_crop),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ])
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        real_idx = int(self.indices[index])
        sample = self.dataset[real_idx]

        image = sample["image"]
        label = sample["label"]  # already 0..99

        multi_crops = [t(image) for t in self.trans]

        if self.return_index:
            return real_idx, multi_crops

        if self.CICL:
            if self.train and not self.val:
                return multi_crops, label
            else:
                return multi_crops[0], label

        return multi_crops


# ------------------------------------------------------------
# Augmentations (UNCHANGED)
# ------------------------------------------------------------
class PILRandomGaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    color_jitter = transforms.ColorJitter(
        0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
    )
    return transforms.Compose([
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ])
