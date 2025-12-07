import argparse
import math
import os
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
# import apex
# from apex.parallel.LARC import LARC

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
)
from src.multicropcifar100 import MultiCropDataset
import src.resnet50 as resnet_models
datestamp = time.strftime("%Y%m%d%H%M%S")

logger = getLogger()

parser = argparse.ArgumentParser(description="Single-GPU SwAV")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2,6], nargs="+")
parser.add_argument("--size_crops", type=int, default=[128,64], nargs="+")
parser.add_argument("--min_scale_crops", type=float, default=[0.14,0.2], nargs="+")
parser.add_argument("--max_scale_crops", type=float, default=[1.0,0.4], nargs="+")

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0])
parser.add_argument("--temperature", default=0.1, type=float)
parser.add_argument("--epsilon", default=0.03, type=float)
parser.add_argument("--sinkhorn_iterations", default=3, type=int)
parser.add_argument("--feat_dim", default=128, type=int)
parser.add_argument("--nmb_prototypes", default=250, type=int)
parser.add_argument("--queue_length", type=int, default=0)
parser.add_argument("--epoch_queue_starts", type=int, default=2)

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=400, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--base_lr", default=0.4, type=float)
parser.add_argument("--final_lr", type=float, default=0)
parser.add_argument("--freeze_prototypes_niters", default=0, type=int)
parser.add_argument("--wd", default=1e-6, type=float)
parser.add_argument("--warmup_epochs", default=10, type=int)
parser.add_argument("--start_warmup", default=0, type=float)

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str)
parser.add_argument("--hidden_mlp", default=2048, type=int)
parser.add_argument("--workers", default=8, type=int)
parser.add_argument("--checkpoint_freq", type=int, default=25)
parser.add_argument("--use_fp16", type=bool_flag, default=True)
parser.add_argument("--dump_path", type=str, default=".")
parser.add_argument("--seed", type=int, default=31)
def extract_train_features(model, train_loader, device='cuda'):
    model.eval()
    feats = []
    labels = []

    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            f,_ = model(x)     # 你要确保 model forward 输出的是 feature, not logits
            feats.append(f.cpu())
            labels.append(y)

    feats = torch.cat(feats, dim=0)      # (50000, D)
    labels = torch.cat(labels, dim=0)    # (50000,)
    return feats, labels

def count_prototype_usage(all_Q_epoch, nmb_prototypes):
    """
    all_Q_epoch: list of Q tensors, each (B, K)
    nmb_prototypes: K
    """
    import torch

    # Concatenate all batches of Q
    Q_all = torch.cat(all_Q_epoch, dim=0)   # (N_total, K)

    # argmax per sample (prototype id)
    assigned = torch.argmax(Q_all, dim=1)   # (N_total,)

    # Count usage
    counts = torch.bincount(assigned, minlength=nmb_prototypes)

    # print sorted from most used to least used
    sorted_idx = torch.argsort(counts, descending=True)

    # print("\n=== Prototype usage (sorted) ===")
    # for idx in sorted_idx:
    #     print(f"Prototype {idx.item():3d}: {counts[idx].item()}")
    auto_bins_by_quantile([counts[i].item() for i in sorted_idx], n_bins=6)


    return counts
import numpy as np

def auto_bins_by_quantile(counts, n_bins=10):
    """
    counts: tensor/list/np.array of prototype usage counts (length K)
    n_bins: how many bins (groups) you want; auto behavior remains dynamic
    """

    counts_np = np.array(counts)

    # 自动根据 quantile 生成区间边界
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(counts_np, quantiles)

    # 避免出现重复边界（比如全 0）
    bin_edges = np.unique(bin_edges)

    print("\n=== Prototype Usage Distribution by Quantile Bins ===")

    # 统计每个区间的数量
    for i in range(len(bin_edges) - 1):
        low = bin_edges[i]
        high = bin_edges[i + 1]

        mask = (counts_np >= low) & (counts_np <= high)
        num_items = np.sum(mask)

        print(f"Range [{low:.1f} – {high:.1f}]: {num_items} prototypes")

    return bin_edges

def knn_eval(train_feats, train_labels, test_feats, test_labels, k=20):
    """
    train_feats: (N, D)
    test_feats:  (M, D)
    """
    import torch
    dist = torch.cdist(test_feats, train_feats)   # (M, N)

    # find k nearest neighbors
    knn_idx = dist.topk(k, largest=False).indices        # (M, k)
    knn_labels = train_labels[knn_idx]                   # (M, k)

    # majority vote
    pred = torch.mode(knn_labels, dim=1).values          # (M,)

    acc = (pred == test_labels).float().mean().item() * 100
    return acc
from torch.utils.data import Dataset
import torchvision
class SubsampledDataset(Dataset):
    def __init__(self, dataset, ratio=0.05, seed=0):
        self.dataset = dataset
        N = len(dataset)
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(N, generator=g)[:int(N * ratio)]
        self.indices = idx.tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[self.indices[i]]
import re
def get_epoch_from_switch(filepath): 
            filepath = "knn_switch_"+filepath
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                line = first_line.strip().lower()

                if "epoch" not in line:
                    print("第一行不包含 'epoch'")
                    return []

                try:
                    # 提取冒号后面的内容
                    raw = line.split(":", 1)[1].strip()

                    # 去掉可能的方括号
                    raw = raw.strip("[]")

                    # 用正则提取所有整数
                    nums = re.findall(r'\d+', raw)

                    # 转成 int 数组
                    epoch_list = [int(n) for n in nums]

                    return epoch_list

                except Exception as e:
                    print("无法解析 epoch 数组:", e)
                    return []


  

def main():
    global args
    args = parser.parse_args()
    with open("knn_switch_"+datestamp, 'w') as f:
        f.write("epoch: [-1]\n")
        f.write(args.__repr__())
        print("file name is knn_switch_"+datestamp)
    # CHANGED: 单机单卡固定 rank/world_size
    args.rank = 0
    args.world_size = 1

    fix_random_seeds(args.seed)
    logger_, training_stats = initialize_exp(args, "epoch", "loss")

    # ============ build data ============
    train_dataset = MultiCropDataset(
        args.data_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,            # CHANGED: 单卡直接 shuffle
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
     
    test_transform = torchvision.transforms.Compose([
       torchvision.transforms.Resize(224),
    torchvision.transforms.CenterCrop(224),  # CIFAR 的固定大小
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5071, 0.4865, 0.4409],
                         [0.2675, 0.2565, 0.2761]),
    ])
    test_dataset = torchvision.datasets.CIFAR100(
        root='./',
        train=False,
        transform=test_transform,
        download=True,
    )
    test_dataset = SubsampledDataset(test_dataset, ratio=0.5, seed=0)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    knn_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5071, 0.4865, 0.4409],
                         [0.2675, 0.2565, 0.2761]),
    ])

    train_knn_dataset = torchvision.datasets.CIFAR100(
        root='./',
        train=True,
        transform=knn_transform,
        download=True,
    )
    train_knn_dataset = SubsampledDataset(train_knn_dataset, ratio=0.25, seed=0)
    train_knn_loader = torch.utils.data.DataLoader(
        train_knn_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    logger.info(f"Building data done with {len(train_dataset)} images loaded.")

    # ============ build model ============
    model = resnet_models.__dict__[args.arch](
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_prototypes=args.nmb_prototypes,
    )

    # CHANGED: 不再做 SyncBatchNorm / DDP
    model = model.cuda()
    logger.info(model)
    logger.info("Building model done.")

    # ============ build optimizer ============
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )
    # optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    warmup_lr_schedule = np.linspace(
        args.start_warmup,
        args.base_lr,
        len(train_loader) * args.warmup_epochs,
    )
    iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
    cosine_lr_schedule = np.array([
        args.final_lr + 0.5 * (args.base_lr - args.final_lr)
        * (1 + math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs))))
        for t in iters
    ])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # ============ init mixed precision ============
    # if args.use_fp16:
    #     model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
    #     logger.info("Initializing mixed precision done.")
    if args.use_fp16:
        from torch.amp import GradScaler, autocast
        scaler = GradScaler("cuda")
    else:
        scaler = None
    # CHANGED: 不再包 DistributedDataParallel
    # model = nn.parallel.DistributedDataParallel(...)

    # ============ optionally resume ============
    to_restore = {"epoch": 0}
    # restart_from_checkpoint(
    #     os.path.join(args.dump_path, "checkpoint.pth.tar"),
    #     run_variables=to_restore,
    #     state_dict=model,
    #     optimizer=optimizer,
    #     amp=None,
    # )
    start_epoch = to_restore["epoch"]

    # ============ build the queue ============
    queue = None
    queue_path = os.path.join(args.dump_path, "queue.pth")   # CHANGED: 不再按 rank 区分
    if os.path.isfile(queue_path):
        queue = torch.load(queue_path)["queue"]
    # 单卡：只需保证能整除 batch_size
    args.queue_length -= args.queue_length % args.batch_size

    cudnn.benchmark = True
    knn_accs = []
    for epoch in range(start_epoch, args.epochs):
        if epoch in get_epoch_from_switch(datestamp) or -1 in get_epoch_from_switch(datestamp):
            train_feats, train_labels = extract_train_features(model, train_knn_loader)
            #print(train_feats.shape, train_labels.shape)
            test_feats, test_labels   = extract_train_features(model, test_loader)
            #print(train_feats.shape, train_labels.shape,test_feats.shape, test_labels.shape)
            acc = knn_eval(train_feats, train_labels, test_feats, test_labels)
            knn_accs.append((epoch, acc))
            print(knn_accs)
        logger.info(f"============ Starting epoch {epoch} ...  knn_switch_{datestamp}============")

        # REMOVED: train_loader.sampler.set_epoch(epoch)

        # optionally starts a queue
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(args.crops_for_assign),
                args.queue_length,
                args.feat_dim,
            ).cuda()

        # train the network
        scores, queue = train(train_loader, model, optimizer,scaler, epoch, lr_schedule, queue)
        training_stats.update(scores)
         

        # save checkpoints (单卡：总是保存)
        # save_dict = {
        #     "epoch": epoch + 1,
        #     "state_dict": model.state_dict(),
        #     "optimizer": optimizer.state_dict(),
        # }
        # if args.use_fp16:
            # save_dict["amp"] = apex.amp.state_dict()
        # torch.save(
        #     save_dict,
        #     os.path.join(args.dump_path, "checkpoint.pth.tar"),
        # )
        # if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
        #     dump_ckp_dir = os.path.join(args.dump_path, "checkpoints")
        #     os.makedirs(dump_ckp_dir, exist_ok=True)
        #     shutil.copyfile(
        #         os.path.join(args.dump_path, "checkpoint.pth.tar"),
        #         os.path.join(dump_ckp_dir, f"ckp-{epoch}.pth"),
        #     )

        # if queue is not None:
        #     torch.save({"queue": queue}, queue_path)

from torch.amp import autocast, GradScaler
def train(train_loader, model, optimizer,scaler, epoch, lr_schedule, queue):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    all_Q_epoch = []
    model.train()
    use_the_queue = False

    end = time.time()
    
    for it, inputs in enumerate(train_loader):
        data_time.update(time.time() - end)

        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:   # 如果没再包 LARC 就用这个
            param_group["lr"] = lr_schedule[iteration]

        # normalize the prototypes
        with torch.no_grad():
            w = model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.prototypes.weight.copy_(w)

        # -------- 这里开始分 FP16 / FP32 前向 + loss --------
        if args.use_fp16:
             
            with autocast(device_type="cuda"     ):
                embedding, output = model(inputs)
                # print("embedding:", embedding.shape)
                # print("output:", output.shape)
                embedding = embedding.detach()
                bs = inputs[0].size(0)

                loss = 0
                for i, crop_id in enumerate(args.crops_for_assign):
                    with torch.no_grad():
                        out = output[bs * crop_id: bs * (crop_id + 1)].detach()
                        if it % 150 == 0 :
                            # 沿 prototype 维度统计
                            logit_std = out.std(dim=1).mean().item ()   # 每个样本的logits标准差，再平均
                            logit_abs_mean = out.abs().mean().item()
                            logger.info(
                                f"[Debug] logits std per sample: {logit_std:.4f}, "
                                f"abs mean: {logit_abs_mean:.4f}"
                            )
                        if queue is not None:
                            if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                                use_the_queue = True
                                out = torch.cat((
                                    torch.mm(queue[i], model.prototypes.weight.t()),
                                    out,
                                ))
                            nmb_crops = np.sum(args.nmb_crops)
                            embedding = embedding.view(nmb_crops, bs, -1).contiguous()
                            crop_emb = embedding[crop_id]
                            queue[i, bs:] = queue[i, :-bs].clone()
                            #queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]
                            queue[i, :bs] = crop_emb
                        q = distributed_sinkhorn(out)[-bs:]
                        with torch.no_grad():
                            all_Q_epoch.append(q.detach().cpu())
                    if it % 150 == 0:
                        #print q.std, q.mean()
                        logger.info(f"Sinkhorn Q: std {q.std().item():.4f}, mean {q.mean().item():.4f}")
                    subloss = 0
                    for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                        x = output[bs * v: bs * (v + 1)] / args.temperature
                        subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                    loss += subloss / (np.sum(args.nmb_crops) - 1)
                loss /= len(args.crops_for_assign)

        else:
            # FP32 前向 + loss（跟上面一模一样，只是没 autocast）
            embedding, output = model(inputs)
            embedding = embedding.detach()
            bs = inputs[0].size(0)

            loss = 0
            for i, crop_id in enumerate(args.crops_for_assign):
                with torch.no_grad():
                    out = output[bs * crop_id: bs * (crop_id + 1)].detach()

                    if queue is not None:
                        if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                            use_the_queue = True
                            out = torch.cat((
                                torch.mm(queue[i], model.prototypes.weight.t()),
                                out,
                            ))
                        queue[i, bs:] = queue[i, :-bs].clone()
                        queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                    q = distributed_sinkhorn(out)[-bs:]
                if it % 100 == 0:
                        #print q.std, q.mean()
                        logger.info(f"Sinkhorn Q: std {q.std().item():.4f}, mean {q.mean().item():.4f}")
                subloss = 0
                for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                    x = output[bs * v: bs * (v + 1)] / args.temperature
                    subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                loss += subloss / (np.sum(args.nmb_crops) - 1)
            loss /= len(args.crops_for_assign)

        # -------- backward + step（统一写法） --------
        optimizer.zero_grad()
        if args.use_fp16:
            scaler.scale(loss).backward()
            if iteration < args.freeze_prototypes_niters:
                for name, p in model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if iteration < args.freeze_prototypes_niters:
                for name, p in model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
            optimizer.step()
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
    count_prototype_usage(all_Q_epoch, args.nmb_prototypes)
    return (epoch, losses.avg), queue


@torch.no_grad()
def distributed_sinkhorn(out):
    """
    单机单卡版 Sinkhorn-Knopp，不再使用 torch.distributed
    """
    Q = torch.exp(out / args.epsilon).t()   # K x B
    B = Q.shape[1]
    K = Q.shape[0]

    # make the matrix sum to 1
    Q /= torch.sum(Q)

    for _ in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # columns sum to 1
    return Q.t()  # B x K
 


if __name__ == "__main__":
    main()
