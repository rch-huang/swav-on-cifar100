import os
import shutil
import random
import numpy as np
from PIL import ImageFilter

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# ------------------------------------------------------------
# TaskSubset (unchanged; matches your CIFAR-100 implementation)
# ------------------------------------------------------------
class TaskSubset(Dataset):
    def __init__(self, base_dataset, selected_indices):
        self.base_dataset = base_dataset
        self.selected_indices = selected_indices

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, i):
        # follow your convention: base_dataset[...] returns multi_crops (or (multi_crops,label))
        # we return only multi_crops here (index 0), same as your CIFAR code
        return self.base_dataset[self.selected_indices[i]][0]


# ------------------------------------------------------------
# MultiCropDataset (Tiny-ImageNet)
# ------------------------------------------------------------
class MultiCropDataset(Dataset):
    def __init__(
        self,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        size_dataset=-1,
        CICL=True,
        return_index=False,
        train=True,
        val = False,
        val_size = 5000,
        seed=0,
        select_100_of_200=True,
        num_used_classes=100,
        num_tasks=10,
        folderized_train_dir='/Scratch/repository/rh539/tiny-imagenet-200/train_foldered',    # default: <root>/train_foldered
        folderized_val_dir='/Scratch/repository/rh539/tiny-imagenet-200/val_foldered',      # default: <root>/val_foldered
        selected_100classes_out_of_200_for_tinyimagenet=None,
        class_map=None,
     ):
        super().__init__()
        self.CICL = CICL
        self.return_index = return_index
        self.train = train
        self.val = val
         

         

        data_root = folderized_train_dir if train else folderized_val_dir

      
        self.dataset = datasets.ImageFolder(
            root=data_root,
            transform=None,  # do multi-crop transform inside
        )
        if selected_100classes_out_of_200_for_tinyimagenet == None:    
            total_classes = len(self.dataset.classes)
            rng = np.random.RandomState(seed)
            all_cls = np.arange(total_classes)
            selected_100classes_out_of_200_for_tinyimagenet = rng.choice(all_cls, size=num_used_classes, replace=False)
            selected_100classes_out_of_200_for_tinyimagenet = [int(x) for x in selected_100classes_out_of_200_for_tinyimagenet]
            classes_per_task = num_used_classes // num_tasks  
         
        self.selected_100classes_out_of_200_for_tinyimagenet = selected_100classes_out_of_200_for_tinyimagenet
         
      

         
        active_classes = set(selected_100classes_out_of_200_for_tinyimagenet)

         
        filtered = []
        for i, (_, y) in enumerate(self.dataset.samples):
            if y in active_classes:
                filtered.append(i)
        self.indices = np.array(filtered, dtype=np.int64)

        if val is True:
            rng = np.random.RandomState(seed)
            rng.shuffle(self.indices)
            self.indices = self.indices[:val_size]
         
        if class_map is  None: 
            self.class_map = {old: new for new, old in enumerate(selected_100classes_out_of_200_for_tinyimagenet)}
        else:
            self.class_map = class_map
        # -------------------------
        # 6) Multi-crop augmentation (same structure as CIFAR file)
        # -------------------------
        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]

        # Tiny-ImageNet uses ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        self.biggest_crop = size_crops[0]
        trans = []
        if train and val is False:
            for i in range(len(size_crops)):
                randomresizedcrop = transforms.RandomResizedCrop(
                    size_crops[i],
                    scale=(min_scale_crops[i], max_scale_crops[i]),
                )
                trans.extend([
                    transforms.Compose([
                        randomresizedcrop,
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.Compose(color_transform),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std),
                    ])
                ] * nmb_crops[i])
        else:
            test_transform = transforms.Compose([
                transforms.Resize(self.biggest_crop),
                transforms.CenterCrop(self.biggest_crop),   
                transforms.ToTensor(),
                transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
            trans.extend([test_transform])
        self.trans = trans
    def full_selected_labels(self):
        # return the full labels array (mapped to selected classes) for the current dataset
        labels_of_selected = [self.dataset.samples[i][1] for i in self.indices]
        mapped_labels = np.array([self.class_map[label] for label in labels_of_selected])
        return mapped_labels
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        # index is in [0, len(self.indices)-1]
        # real_idx is an index into the underlying ImageFolder dataset
        real_idx = int(self.indices[index])
        image, label = self.dataset[real_idx]

        # multi-crop views
        multi_crops = [t(image) for t in self.trans]

        # remap label into contiguous space if desired (0..99)
        if self.class_map is not None:
            label = self.class_map[label]

        if self.return_index:
            return real_idx, multi_crops
        if self.CICL:
            if self.train and self.val is False:
                return multi_crops, label
            else:
                return multi_crops[0], label
        return multi_crops


# ------------------------------------------------------------
# PIL blur + color distortion (same as CIFAR file)
# ------------------------------------------------------------
class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    Used in SimCLR - https://arxiv.org/abs/2002.05709
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
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

# -------------------------
#  Folder preparation helpers
# -------------------------
def prepare_tinyimagenet_train_as_imagefolder(root, out_dir=None, overwrite=False):
    """
    Official tiny-imagenet-200 train structure is:
        train/<wnid>/images/*.JPEG
    ImageFolder expects:
        train_foldered/<wnid>/*.JPEG

    This helper creates a foldered copy via hardlink (if possible) or copy.

    Args:
        root: tiny-imagenet-200 path
        out_dir: output directory path (default: root/train_foldered)
        overwrite: if True, remove and rebuild out_dir

    Returns:
        out_dir
    """
    train_root = os.path.join(root, "train")
    if out_dir is None:
        out_dir = os.path.join(root, "train_foldered")

    if overwrite and os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    # each class: train/<wnid>/images/*.JPEG
    for wnid in os.listdir(train_root):
        class_dir = os.path.join(train_root, wnid)
        images_dir = os.path.join(class_dir, "images")
        if not os.path.isdir(images_dir):
            continue

        dst_class_dir = os.path.join(out_dir, wnid)
        os.makedirs(dst_class_dir, exist_ok=True)

        for fn in os.listdir(images_dir):
            if not fn.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            src = os.path.join(images_dir, fn)
            dst = os.path.join(dst_class_dir, fn)

            if os.path.exists(dst):
                continue
            try:
                os.link(src, dst)  # hardlink to save space
            except OSError:
                shutil.copy2(src, dst)

    return out_dir


def prepare_tinyimagenet_val_as_imagefolder(root, out_dir=None, overwrite=False):
    """
    Official tiny-imagenet-200 val structure is:
        val/images/*.JPEG
        val/val_annotations.txt  (img_name \\t wnid \\t x \\t y \\t w \\t h)

    ImageFolder expects:
        val_foldered/<wnid>/*.JPEG

    This helper creates that foldered val via hardlink (if possible) or copy.

    Args:
        root: tiny-imagenet-200 path
        out_dir: output directory path (default: root/val_foldered)
        overwrite: if True, remove and rebuild out_dir

    Returns:
        out_dir
    """
    val_root = os.path.join(root, "val")
    images_dir = os.path.join(val_root, "images")
    ann_file = os.path.join(val_root, "val_annotations.txt")

    if out_dir is None:
        out_dir = os.path.join(root, "val_foldered")

    if overwrite and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # parse annotations
    mapping = {}
    with open(ann_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            img_name, wnid = parts[0], parts[1]
            mapping[img_name] = wnid

    for img_name, wnid in mapping.items():
        src = os.path.join(images_dir, img_name)
        if not os.path.exists(src):
            continue
        dst_class_dir = os.path.join(out_dir, wnid)
        os.makedirs(dst_class_dir, exist_ok=True)
        dst = os.path.join(dst_class_dir, img_name)

        if os.path.exists(dst):
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    return out_dir


# -------------------------
#  Minimal usage example
# -------------------------
if __name__ == "__main__":
    # Example usage (adjust args / paths to your project):
    class Args:
        size_crops = [96, 64]
        nmb_crops = [2, 6]
        min_scale_crops = [0.14, 0.05]
        max_scale_crops = [1.0, 0.14]

    args = Args()

    root = "/Scratch/repository/rh539/tiny-imagenet-200"

    # Prepare foldered train/val for ImageFolder (recommended)
    train_foldered = '/Scratch/repository/rh539/tiny-imagenet-200/train_foldered'
    #prepare_tinyimagenet_train_as_imagefolder(root, overwrite=False)
    val_foldered = '/Scratch/repository/rh539/tiny-imagenet-200/val_foldered'
    #prepare_tinyimagenet_val_as_imagefolder(root, overwrite=False)

    # Create tasks: select 100/200 and split to 10 tasks
    selected_100classes_out_of_200_for_tinyimagenet, tasks = make_tinyimagenet_tasks_100_of_200(seed=42)

    # Build base multicrop dataset (train split)
    # IMPORTANT: point root to the foldered train if you prepared it
    base = MultiCropDataset(
        root=os.path.dirname(train_foldered),   # base root; we will pass train=True and set data_root accordingly
        size_crops=args.size_crops,
        nmb_crops=args.nmb_crops,
        min_scale_crops=args.min_scale_crops,
        max_scale_crops=args.max_scale_crops,
        CICL=True,
       
        selected_100classes_out_of_200_for_tinyimagenet=selected_100classes_out_of_200_for_tinyimagenet,
        val_foldered_root=val_foldered,
    )

    # IMPORTANT: since we used train_foldered, we need to re-point dataset root if needed
    # In this minimal example, we recommend directly using train_foldered by setting:
    #   root=train_foldered's parent and data_root resolving to root/train_foldered
    # If you want strict control, just set:
    #   base.dataset = datasets.ImageFolder(train_foldered, transform=None)
    # and recompute base.indices similarly.

    # Per-task subset indices in base.dataset sample space
    for tid in range(10):
        task_classes = tasks[tid]
        task_indices = indices_for_classes(base, task_classes)
        task_ds = TaskSubset(base, task_indices)
        print(f"Task {tid}: {len(task_ds)} samples, classes={task_classes[:3]}...")
