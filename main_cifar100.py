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
from src.multicropcifar100 import MultiCropDataset, TaskSubset
import src.resnet50 as resnet_models
datestamp = time.strftime("%Y%m%d%H%M%S")

logger = getLogger()

parser = argparse.ArgumentParser(description="Single-GPU SwAV")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2,4], nargs="+")
parser.add_argument("--size_crops", type=int, default=[128,64], nargs="+")
parser.add_argument("--min_scale_crops", type=float, default=[0.14,0.2], nargs="+")
parser.add_argument("--max_scale_crops", type=float, default=[1.0,0.4], nargs="+")

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0])
parser.add_argument("--temperature", default=0.1, type=float)
parser.add_argument("--epsilon", default=0.05, type=float)
parser.add_argument("--sinkhorn_iterations", default=3, type=int)
parser.add_argument("--feat_dim", default=128, type=int)
parser.add_argument("--nmb_prototypes", default=250, type=int)
parser.add_argument("--queue_length", type=int, default=1000)
parser.add_argument("--epoch_queue_starts", type=int, default=0)

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
parser.add_argument("--workers", default=0, type=int)
parser.add_argument("--checkpoint_freq", type=int, default=25)
parser.add_argument("--use_fp16", type=bool_flag, default=True)
parser.add_argument("--dump_path", type=str, default=".")
parser.add_argument("--seed", type=int, default=30)
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

def count_prototype_usage(all_Q_epoch, nmb_prototypes,epoch=-1,logfile='train_log.txt'):
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
    auto_bins_by_quantile([counts[i].item() for i in sorted_idx], n_bins=6,epoch=epoch,logfile=logfile)
     

    return counts
import numpy as np

def auto_bins_by_quantile(counts, n_bins=10,epoch=-1,logfile='train_log.txt'):
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
        with open(logfile, 'a') as f:
            f.write(f"Epoch {epoch} Prototype Usage Range [{low:.1f} – {high:.1f}]: {num_items} prototypes\n")  
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
def get_knneval_epoch_from_switch(_datestamp): 
            filepath = "knn_switch_"+_datestamp
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                line = first_line.strip().lower()

                if "epoch" not in line:
                    
                    return []

                try:
                     
                    raw = line.split(":", 1)[1].strip()

                    
                    raw = raw.strip("[]")

                     
                    nums = re.findall(r'\d+', raw)

                    
                    epoch_list = [int(n) for n in nums]

                    return epoch_list

                except Exception as e:
                    print(  e)
                    return []
def build_probe_loader(args):
    """
    从训练集里固定抽取一小部分样本（默认 512），
    用一个**确定性**的弱增广（类似 test_transform）来做 probe。

    注意：
    - shuffle=False，保证每个 epoch 样本顺序一致，
    - transform 没有随机性（Resize + CenterCrop），保证同一张图每次 crop 一致。
    """
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader, Subset
    import torch

    probe_transform = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
    ])

    if True:
        base_dataset = datasets.CIFAR100(
            root='./',
            train=True,
            transform=probe_transform,
            download=True,
        )

    num_samples = min(512, len(base_dataset))

    # 用 seed 固定下来这一批样本
    g = torch.Generator()
    g.manual_seed(args.seed + 1234)
    indices = torch.randperm(len(base_dataset), generator=g)[:num_samples]

    probe_dataset = Subset(base_dataset, indices.tolist())

    probe_loader = DataLoader(
        probe_dataset,
        batch_size=args.batch_size,
        shuffle=False,              # 很重要：保证每个 epoch 顺序一致
        num_workers=args.workers,
        pin_memory=True,
    )
    return probe_loader

def get_save_epoch_from_switch(_filepath): 
            filepath = "knn_switch_"+_filepath
            with open(filepath, 'r', encoding='utf-8') as f:
                 
                first_line = f.readline().strip()
                second_line = f.readline().strip() 
                third_line = f.readline().strip() 
                line = third_line.strip().lower()

                if "save_epoch" not in line:
                    
                    return []

                try:
                    
                    raw = line.split(":", 1)[1].strip()

                    
                    raw = raw.strip("[]")

                    
                    nums = re.findall(r'\d+', raw)

                     
                    epoch_list = [int(n) for n in nums]

                    return epoch_list

                except Exception as e:
                    print(  e)
                    return []

  
@torch.no_grad()
def compute_probe_kl(model, probe_loader, args, prev_Q=None, device="cuda"):
    """
    使用固定的 probe_loader 计算：
      1) KL(Q || P)，其中 P 是 softmax(logits) 得到的 prototype similarity 分布
      2) KL(Q_t || Q_{t-1})，其中 Q_t 来自当前 epoch 的 Sinkhorn，Q_{t-1} 来自上一 epoch

    返回：
      - kl_q_p:      标量，KL(Q || P)
      - kl_q_t_prev: 标量，KL(Q_t || Q_{t-1})，如果 prev_Q 为空则为 NaN
      - Q_all:       当前 epoch probe 样本对应的 Q_t (N_probe x K) 用于下一个 epoch
    """
    import torch
    import torch.nn.functional as F

    model.eval()

    all_P = []
    all_Q = []

    for inputs, _ in probe_loader:
        inputs = inputs.to(device, non_blocking=True)
        # forward: 与 extract_train_features 一样，单 view 前向
        feats, logits = model(inputs)   # logits: [B, K]

        # P: prototype similarity 的 softmax 分布
        #   这里直接对 logits softmax，就相当于用 "prototype scores" 归一化
        P = F.softmax(logits, dim=1)    # [B, K]

        # Q: 用和训练时相同的 Sinkhorn 算法（但不使用 queue）
        Q = distributed_sinkhorn(logits)  # [B, K]

        all_P.append(P)
        all_Q.append(Q)

    P_all = torch.cat(all_P, dim=0)   # [N_probe, K]
    Q_all = torch.cat(all_Q, dim=0)   # [N_probe, K]

    eps = 1e-6
    P_clamped = P_all.clamp(min=eps)
    Q_clamped = Q_all.clamp(min=eps)


    if False:
          
          
        row_P = P_all.sum(dim=1)
        row_Q = Q_all.sum(dim=1)
        print(f"[Sanity] Row-sum P: mean={row_P.mean():.6f}, std={row_P.std():.6f}")
        print(f"[Sanity] Row-sum Q: mean={row_Q.mean():.6f}, std={row_Q.std():.6f}")

        # Column-sum of Q
        col_Q = Q_all.sum(dim=0)
        print(f"[Sanity] Column-sum Q: mean={col_Q.mean():.6f}, std={col_Q.std():.6f}")
        print(f"[Sanity] Column-sum Q range: [{col_Q.min():.6f}, {col_Q.max():.6f}]")

        # Difference
        print(f"[Sanity] Mean |Q-P| = {(Q_all - P_all).abs().mean().item():.6f}")

        # Manual KL
        log_Q = Q_clamped.log()
        log_P = P_clamped.log()
        kl_manual = (Q_clamped * (log_Q - log_P)).sum(dim=1).mean()
        print(f"[Sanity] KL manual (Q||P) = {kl_manual.item():.6f}")

        # Q||Q_prev
        if prev_Q is not None:
            prev_Q_clamped = prev_Q.to(device).clamp(min=eps)
            kl_q_prev_manual = (Q_clamped * (log_Q - prev_Q_clamped.log())).sum(dim=1).mean()
            print(f"[Sanity] KL(Q_t || Q_t-1) = {kl_q_prev_manual.item():.6f}")



    # KL(Q || P) = sum_k Q_k log(Q_k / P_k)
    kl_q_p = (Q_clamped * (Q_clamped.log() - P_clamped.log())).sum(dim=1).mean()

    # KL(Q_t || Q_{t-1})
    if prev_Q is not None:
        prev_Q = prev_Q.to(device)
        prev_Q_clamped = prev_Q.clamp(min=eps)
        kl_q_t_prev = (Q_clamped * (Q_clamped.log() - prev_Q_clamped.log())).sum(dim=1).mean()
    else:
        kl_q_t_prev = torch.tensor(float("nan"), device=device)

    model.train()
    # 返回两个 scalar + 当前的 Q_all，方便外面保存
    return kl_q_p.item(), kl_q_t_prev.item(), Q_all.detach().cpu()

def main():
    global args
    args = parser.parse_args()
    with open("knn_switch_"+datestamp, 'w') as f:
        f.write("epoch: [1,3,5,10]\n")
        f.write(args.__repr__()+"\n")
        f.write("save_epoch: [3,10]\n")
        print("file name is knn_switch_"+datestamp)
    
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
    train_loaders = []
    train_loader = None
    CICL = False
    if CICL:
        full_labels = np.array(train_dataset.dataset.targets)  # shape=(50000,)

        num_tasks = 10
        classes_per_task = 10

        task_class_lists = [
            list(range(t * classes_per_task, (t + 1) * classes_per_task))
            for t in range(num_tasks)
        ]
        task_indices_list = []

        for class_list in task_class_lists:
            mask = np.isin(full_labels, class_list)
            indices = np.where(mask)[0]
            task_indices_list.append(indices)
        
         

        for t, indices in enumerate(task_indices_list):
            subset = TaskSubset(train_dataset, indices)
            loader = torch.utils.data.DataLoader(
                subset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=True,
            )
            train_loaders.append(loader)
            print(f"Task {t}: classes {task_class_lists[t]}, samples={len(indices)}")
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,            
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
        train_loaders.append(train_loader)
    biggest_crop = args.size_crops[0]
    test_transform = torchvision.transforms.Compose([
       torchvision.transforms.Resize(biggest_crop),
    torchvision.transforms.CenterCrop(biggest_crop),   
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
        torchvision.transforms.Resize(biggest_crop),
        torchvision.transforms.CenterCrop(biggest_crop),
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
    probe_loader = build_probe_loader(args)
    probe_Q_prev = None
    logger.info(f"Building data done with {len(train_dataset)} images loaded.")

    # ============ build model ============
    model = resnet_models.__dict__[args.arch](
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_prototypes=args.nmb_prototypes,
    )

    
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
    # queue_path = os.path.join(args.dump_path, "queue.pth")   # CHANGED: 不再按 rank 区分
    # if os.path.isfile(queue_path):
    #     queue = torch.load(queue_path)["queue"]
     
    args.queue_length -= args.queue_length % args.batch_size

    cudnn.benchmark = True
    knn_accs = []
    logfile = 'train_log_'+datestamp
    for epoch in range(start_epoch, args.epochs):
        if epoch in get_knneval_epoch_from_switch(datestamp) or -1 in get_knneval_epoch_from_switch(datestamp):
            train_feats, train_labels = extract_train_features(model, train_knn_loader)
            #print(train_feats.shape, train_labels.shape)
            test_feats, test_labels   = extract_train_features(model, test_loader)
            #print(train_feats.shape, train_labels.shape,test_feats.shape, test_labels.shape)
            acc = knn_eval(train_feats, train_labels, test_feats, test_labels)
            with open(logfile, 'a') as f:
                f.write(f"Epoch {epoch} KNN eval accuracy: {acc:.2f}%\n")
            knn_accs.append((epoch, acc))
            print(knn_accs)
        logger.info(f"============ Starting epoch {epoch} ...  knn_switch_{datestamp}   ============")

        # REMOVED: train_loader.sampler.set_epoch(epoch)

        # optionally starts a queue
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            queue = torch.zeros(
                len(args.crops_for_assign),
                args.queue_length,
                args.feat_dim,
            ).cuda()
         
        # train the network
        scores, queue = train(train_loader, 
                                    model,
                                    optimizer,
                                    scaler,
                                    epoch,
                                    lr_schedule,
                                    queue,
                                    logfile=logfile)
        training_stats.update(scores)
        kl_q_p, kl_q_t_prev, probe_Q_prev = compute_probe_kl(
            model,
            probe_loader,
            args,
            prev_Q=probe_Q_prev,
            device="cuda",
        ) 
        logger.info(
            f"[Probe KL] Epoch {epoch} "
            f"KL(Q || P) = {kl_q_p:.4f}, KL(Q_t || Q_(t-1)) = {kl_q_t_prev:.4f}"
        )
        with open(logfile, 'a') as f:
            f.write(
                f"Epoch {epoch} Probe KL(Q || P) = {kl_q_p:.4f}, KL(Q_t || Q_(t-1)) = {kl_q_t_prev:.4f}\n"
            )
        if epoch in get_save_epoch_from_switch(datestamp) or -1 in get_save_epoch_from_switch(datestamp):
            
            print("Saving checkpoint at epoch ", epoch)
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if args.use_fp16:
                # save_dict["amp"] = apex.amp.state_dict()
                save_dict["scaler"] = scaler.state_dict()
            path = os.path.join('', f"checkpoint_datestamp_{datestamp}_epoch_{epoch}.pth.tar")
            torch.save(
                save_dict,
                path,
            )
         
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
def train(train_loader, model, optimizer,scaler, epoch, lr_schedule, queue,logfile='train_log.txt'):
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
        for param_group in optimizer.param_groups:   
            param_group["lr"] = lr_schedule[iteration]

        # normalize the prototypes
        with torch.no_grad():
            w = model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.prototypes.weight.copy_(w)

         
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
                        if it % 50 == 0 :
                             
                            logit_std = out.std(dim=1).mean().item ()   
                            logit_abs_mean = out.abs().mean().item()
                            logger.info(
                                f"[Debug] logits std per sample: {logit_std:.4f}, "
                                f"abs mean: {logit_abs_mean:.4f}"
                            )
                            with open(logfile, 'a') as f:
                                f.write(
                                    f"Epoch {epoch} Iter {it} Logit std per sample: {logit_std:.4f}, "
                                    f"abs mean: {logit_abs_mean:.4f}\n"
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
                    if it % 50 == 0:
                        #print q.std, q.mean()
                        logger.info(f"Sinkhorn Q: std {q.std().item():.4f}, mean {q.mean().item():.4f}")
                        with open(logfile, 'a') as f:
                            f.write(f"Epoch {epoch} Iter {it} Sinkhorn Q: std {q.std().item():.4f}, mean {q.mean().item():.4f}\n")   

                    subloss = 0
                    for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                        x = output[bs * v: bs * (v + 1)] / args.temperature
                        subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                    loss += subloss / (np.sum(args.nmb_crops) - 1)
                loss /= len(args.crops_for_assign)

        else:
             
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
            with open(logfile, 'a') as f:
                f.write(f"Epoch {epoch} Iter {it} Loss: {loss.item():.4f}\n")
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
    count_prototype_usage(all_Q_epoch, args.nmb_prototypes,epoch=epoch, logfile=logfile)
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
