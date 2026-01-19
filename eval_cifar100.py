import argparse
import os
import time
from logging import getLogger

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from src.utils import (
    bool_flag,
    initialize_exp,
    fix_random_seeds,
    AverageMeter,
    accuracy,
)
import src.resnet50 as resnet_models   # <-- your ResNet( BasicBlock=[2,2,2,2] )

logger = getLogger()

parser = argparse.ArgumentParser(description="Linear eval for SwAV CIFAR100 backbone")

parser.add_argument("--data_path", type=str, default="./",
                    help="path to CIFAR100")
parser.add_argument("--dump_path", type=str, default="./linear_eval")
parser.add_argument("--workers", default=4, type=int)
parser.add_argument("--pretrained", default="", type=str)

parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--lr", default=0.3, type=float)
parser.add_argument("--wd", default=1e-6, type=float)

def main():
    global args, best_acc
    args = parser.parse_args()
    fix_random_seeds(31)

     

    # ===============================
    # 1. CIFAR100 dataset
    # ===============================
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4865, 0.4409],
        std=[0.2675, 0.2565, 0.2761],
    )

    train_transform = transforms.Compose([
        transforms.Resize(96),
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        normalize,
    ])


     
    train_set = datasets.CIFAR100(
        args.data_path, train=True, download=True, transform=train_transform)
    val_set = datasets.CIFAR100(
        args.data_path, train=False, download=True, transform=val_transform)
    if False:
        targets_train = np.array(train_set.targets)
        targets_val = np.array(val_set.targets)

        subset_classes = list(range(10))  # task0
        class_map = {c:i for i,c in enumerate(subset_classes)}  # eg. 10->0, 12->1, ...

        subset_idx_train = np.where(np.isin(targets_train, subset_classes))[0]
        subset_idx_val   = np.where(np.isin(targets_val, subset_classes))[0]

        train_set = SubsetWithLabelMap(train_set, subset_idx_train, class_map)
        val_set   = SubsetWithLabelMap(val_set, subset_idx_val, class_map)
      
         
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print("CIFAR100 loaded.")

    # ===============================
    # 2. Build backbone (eval_mode=True)
    # ===============================
    backbone = resnet_models.resnet50(eval_mode=True,
                                      normalize=False,
                                      output_dim=0,
                                      hidden_mlp=0,
                                      nmb_prototypes=0)
    backbone = backbone.cuda()
    backbone.eval()

    # ===============================
    # 3. Load your SwAV checkpoint
    # ===============================
    ckpt = torch.load(args.pretrained, map_location="cuda",weights_only=False)
    state_dict = ckpt["state_dict"]   # <-- EXACTLY as your save_dict
    backbone.load_state_dict(state_dict, strict=False)
    print("Loaded pretrained backbone.")

    # ===============================
    # 4. Linear head (input 512, output 100)
    # ===============================
    classifier = nn.Linear(512, 100).cuda()
    classifier.weight.data.normal_(0, 0.01)
    classifier.bias.data.zero_()

    optimizer = torch.optim.SGD(
        classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
    )

    cudnn.benchmark = True
    best_acc = 0.0

    # ===============================
    # 5. Training loop
    # ===============================
    for epoch in range(args.epochs):
        print(f"=== Epoch {epoch} ===")
        scores = train_one_epoch(backbone, classifier, optimizer, train_loader)
        val_scores = validate(backbone, classifier, val_loader)
         

        # Save checkpoint
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        os.makedirs(args.dump_path, exist_ok=True)
        torch.save(save_dict, os.path.join(args.dump_path, "classifier_ckpt.pth"))

    print(f"Done. Best top1 = {best_acc:.2f}")


def extract_backbone_features(backbone, x):
    """Call only the backbone feature extractor => 512-d vector"""
    feat_map = backbone.forward_backbone(x)
    feat = backbone.avgpool(feat_map).flatten(1)
    return feat


def train_one_epoch(backbone, classifier, optimizer, loader):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    backbone.eval()
    classifier.train()
    criterion = nn.CrossEntropyLoss().cuda()

    end = time.time()

    for imgs, labels in loader:
        imgs = imgs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        with torch.no_grad():
            feats = extract_backbone_features(backbone, imgs)

        out = classifier(feats)
        loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(out, labels, topk=(1, 5))
        losses.update(loss.item(), imgs.size(0))
        top1.update(acc1[0], imgs.size(0))
        top5.update(acc5[0], imgs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, top1.avg.item(), top5.avg.item()


def validate(backbone, classifier, loader):
    global best_acc

    backbone.eval()
    classifier.eval()
    criterion = nn.CrossEntropyLoss().cuda()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            feats = extract_backbone_features(backbone, imgs)
            out = classifier(feats)

            loss = criterion(out, labels)
            acc1, acc5 = accuracy(out, labels, topk=(1, 5))

            losses.update(loss.item(), imgs.size(0))
            top1.update(acc1[0], imgs.size(0))
            top5.update(acc5[0], imgs.size(0))

    if top1.avg.item() > best_acc:
        best_acc = top1.avg.item()

    print(f"Val top1={top1.avg:.2f}, best={best_acc:.2f}")
    return losses.avg, top1.avg.item(), top5.avg.item()


if __name__ == "__main__":
    main()

