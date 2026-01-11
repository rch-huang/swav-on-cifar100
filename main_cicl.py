import argparse
import math
import os
import shutil
import time
from logging import getLogger
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
# import apex
# from apex.parallel.LARC import LARC
CICL = True
from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
)
import src.resnet50 as resnet_models
datestamp = time.strftime("%Y%m%d%H%M%S")

logger = getLogger()

parser = argparse.ArgumentParser(description="Single-GPU SwAV")

#########################
#### data parameters ####
#########################
 
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+")
parser.add_argument("--size_crops", type=int, default=[128], nargs="+")
parser.add_argument("--min_scale_crops", type=float, default=[0.4 ], nargs="+")
parser.add_argument("--max_scale_crops", type=float, default=[1.0 ], nargs="+")

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0])
parser.add_argument("--temperature", default=0.2, type=float)
parser.add_argument("--epsilon", default=0.03, type=float)
parser.add_argument("--sinkhorn_iterations", default=10, type=int)
parser.add_argument("--feat_dim", default=128, type=int)
parser.add_argument("--nmb_prototypes", default=250, type=int)
parser.add_argument("--queue_length", type=int, default=0)
parser.add_argument("--epoch_queue_starts", type=int, default=5)
parser.add_argument("--queue_cleared_each_task", type=bool_flag, default=True)

#########################
#### optim parameters ###
#########################
parser.add_argument("--dataset", default="tinyimagenet", type=str, choices=["cifar100","tinyimagenet"])    
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--base_lr", default=0.05, type=float)
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
 
parser.add_argument("--use_fp16", type=bool_flag, default=True)
parser.add_argument("--dump_path", type=str, default=".")
parser.add_argument("--seed", type=int, default=40)
parser.add_argument("--clamp_min", type=float, default=-50.0)
parser.add_argument("--selected_100classes_out_of_200_for_tinyimagenet", type=str, default=None)
parser.add_argument("--label_remap_for_tinyimagenet", type=str, default=None)
class SinkhornTracker:
    """
    Track critical statistics for Sinkhorn normalization degeneracy (B-type).
    No logging inside, only keeps running minima.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.min_row_sum = float("inf")
        self.min_col_sum = float("inf")

    @torch.no_grad()
    def update(self, Q):
        # Q: [K, B] inside distributed_sinkhorn
        row_sum = Q.sum(dim=1)   # [K]
        col_sum = Q.sum(dim=0)   # [B]

        rmin = row_sum.min().item()
        cmin = col_sum.min().item()

        if rmin < self.min_row_sum:
            self.min_row_sum = rmin
        if cmin < self.min_col_sum:
            self.min_col_sum = cmin

def extract_train_features(model, train_loader, seen_classes,device='cuda'):
    model.eval()
    feats = []
    labels = []

    with torch.no_grad():
        for x, y in train_loader:
            # filter out unseen classes
            mask = torch.tensor([label in seen_classes for label in y])
            x = x[mask]
            y = y[mask]

            x = x.to(device)
            f,_ = model(x)      
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

   
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(counts_np, quantiles)

    
    bin_edges = np.unique(bin_edges)

    print("\n=== Prototype Usage Distribution by Quantile Bins ===")

     
    for i in range(len(bin_edges) - 1):
        low = bin_edges[i]
        high = bin_edges[i + 1]

        mask = (counts_np >= low) & (counts_np <= high)
        num_items = np.sum(mask)

        print(f"Range [{low:.1f} – {high:.1f}]: {num_items} prototypes")
        with open(logfile, 'a') as f:
            f.write(f"Epoch {epoch} Prototype Usage Range [{low:.1f} – {high:.1f}]: {num_items} prototypes\n")  
    return bin_edges

def knn_eval(train_feats, train_labels, test_feats, test_labels, k=50):
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
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader, Subset
    import torch

     

    if args.dataset == "tinyimagenet":
        from src.multicroptinyimagenet import MultiCropDataset
        base_dataset = MultiCropDataset(
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            CICL=True,
            selected_100classes_out_of_200_for_tinyimagenet=args.selected_100classes_out_of_200_for_tinyimagenet,
            class_map=args.label_remap_for_tinyimagenet,
            train=True,
            val = True
        )
        probe_transform = transforms.Compose([
            transforms.Resize(args.size_crops[0]), 
            transforms.CenterCrop(args.size_crops[0]),
            transforms.ToTensor(),
        ])

    elif args.dataset == "cifar100":
        base_dataset = datasets.CIFAR100(
            root='./',
            train=True,
            transform=probe_transform,
            download=True,
        )
        probe_transform = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
    ])

    num_samples = min(512, len(base_dataset))

 
    g = torch.Generator()
    g.manual_seed(args.seed + 1234)
    indices = torch.randperm(len(base_dataset), generator=g)[:num_samples]

    probe_dataset = Subset(base_dataset, indices.tolist())

    probe_loader = DataLoader(
        probe_dataset,
        batch_size=args.batch_size,
        shuffle=False,              
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

  
import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def compute_probe_kl(model, probe_loader, args, prev_Q=None, device="cuda"):
    model.eval()

    all_logP = []
    all_Q = []

    # 关掉 autocast，避免 fp16 把你的 P 搞出 0/NaN
    with torch.cuda.amp.autocast(False):
        for inputs, _ in probe_loader:
            inputs = inputs.to(device, non_blocking=True)

            feats, logits = model(inputs)          # logits: [B, K]
            logits = logits.float()                # 强制 fp32

            # 让 P 和训练一致（如果训练里用 logits/temperature）
            logP = F.log_softmax(logits / args.temperature, dim=1)  # fp32, stable

            Q = distributed_sinkhorn(logits)        # 你已在 sinkhorn 内 out.float() 更好；这里也确保 logits fp32

            # 先检查 Q 是否 finite（否则后面都没意义）
            if not torch.isfinite(Q).all():
                raise ValueError("Non-finite Q in probe_kl")

            all_logP.append(logP)
            all_Q.append(Q.float())

    logP_all = torch.cat(all_logP, dim=0)           # [N, K] fp32
    Q_all    = torch.cat(all_Q, dim=0)              # [N, K] fp32

    # clamp 只对 logQ 需要；logP 已经是 log-softmax，不会 -inf
    eps = 1e-12
    logQ = torch.log(Q_all.clamp_min(eps))

    kl_q_p = (Q_all * (logQ - logP_all)).sum(dim=1).mean()

    if not torch.isfinite(kl_q_p):
        # 给你更有用的诊断
        q_min = Q_all.min().item()
        q_zero = (Q_all == 0).float().mean().item()
        lp_min = logP_all.min().item()
        raise ValueError(f"KL(Q||P) non-finite. Q_min={q_min:.3e}, Q_zero_frac={q_zero:.3e}, logP_min={lp_min:.3e}")

    # KL(Q_t || Q_{t-1})
    if prev_Q is not None:
        prev_Q = prev_Q.to(device).float()
        logPrev = torch.log(prev_Q.clamp_min(eps))
        kl_q_t_prev = (Q_all * (logQ - logPrev)).sum(dim=1).mean()
        if not torch.isfinite(kl_q_t_prev):
            kl_q_t_prev = torch.tensor(float("nan"), device=device)
    else:
        kl_q_t_prev = torch.tensor(float("nan"), device=device)

    model.train()
    return kl_q_p.item(), kl_q_t_prev.item(), Q_all.detach().cpu()


def main():
    global args
    args = parser.parse_args()
    with open("knn_switch_"+datestamp, 'w') as f:
        f.write("epoch: [1,30,60,90,99]\n")
        f.write(args.__repr__()+"\n")
        f.write("save_epoch: [50,99]\n")
        print("file name is knn_switch_"+datestamp)
    
    args.rank = 0
    args.world_size = 1

    fix_random_seeds(args.seed)
    logger_, training_stats = initialize_exp(args, "epoch", "loss")
    # ============ build data ============
    if args.dataset == "cifar100":
        from src.multicropcifar100 import MultiCropDataset, TaskSubset
        train_dataset = MultiCropDataset(
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            CICL=CICL
        )
    if args.dataset == "tinyimagenet":
        from src.multicroptinyimagenet import MultiCropDataset,TaskSubset 
        train_dataset = MultiCropDataset(
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            CICL=CICL,
         
        )
        args.selected_100classes_out_of_200_for_tinyimagenet = train_dataset.selected_100classes_out_of_200_for_tinyimagenet
        args.label_remap_for_tinyimagenet = train_dataset.class_map
    train_loaders = []
    train_loader = None
    task_class_lists = []
     
    if CICL:
        if args.dataset == 'cifar100':
            full_labels = np.array(train_dataset.dataset.targets)  # shape=(50000,)
        elif args.dataset == 'tinyimagenet':
            full_labels =  train_dataset.full_selected_labels()  # shape=(100000,)
        num_tasks = 10
        classes_per_task = 10

        rng = np.random.default_rng(seed=args.seed)    
        if True:
            all_classes = np.arange(100)
            rng.shuffle(all_classes)
            all_classes = [int(c) for c in all_classes]
            task_class_lists = [
                list(all_classes[t*classes_per_task : (t+1)*classes_per_task])
                for t in range(num_tasks)
            ]
        if False:
             
            task_class_lists[1] = task_class_lists[0]
            
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
        task_class_lists.append(list(range(100)))  # all classes
    biggest_crop = args.size_crops[0]
    if args.dataset == "tinyimagenet":
        from src.multicroptinyimagenet import MultiCropDataset     
        test_dataset = MultiCropDataset(
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            CICL=CICL,
            selected_100classes_out_of_200_for_tinyimagenet=args.selected_100classes_out_of_200_for_tinyimagenet,
            class_map=args.label_remap_for_tinyimagenet,
            train=False
        )
        train_knn_dataset = MultiCropDataset(
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            CICL=CICL,
            selected_100classes_out_of_200_for_tinyimagenet=args.selected_100classes_out_of_200_for_tinyimagenet,
            class_map=args.label_remap_for_tinyimagenet,
            train=True,
            val = True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        ) 
        train_knn_loader = torch.utils.data.DataLoader(
            train_knn_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
    if args.dataset == "cifar100":
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
        train_knn_dataset = SubsampledDataset(train_knn_dataset, ratio=0.5, seed=0)
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

    logfile = 'train_log_'+datestamp
    with open(logfile, 'a') as f:
        f.write(args.__repr__()+"\n")
    model = model.cuda()
    logger.info(model)
    logger.info("Building model done.")
    seen_classes = []
    knn_accs = []
    for _task_number in range(len(train_loaders)):
        task_number = _task_number
        if False:
            if _task_number == 0:
                task_number = 1
            elif _task_number ==1:
                task_number = 0
        train_loader = train_loaders[task_number]
        seen_classes.append(task_class_lists[task_number])
        print(f"Starting training on task {task_number} with classes {task_class_lists[task_number]}...")
        with open(logfile, 'a') as f:
            f.write(f"Starting training on task {task_number} with classes {task_class_lists[task_number]}...\n")
            
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
        

        

        # ============ build the queue ============
        if args.queue_cleared_each_task:
            queue  = None
        else:
            if _task_number == 0:
                queue = None
        
        
        args.queue_length -= args.queue_length % args.batch_size

        cudnn.benchmark = True
        sinkhorn_tracker = SinkhornTracker()


        for epoch in range(0, args.epochs):
            proto_hash_ref = tensor_hash(model.prototypes.weight.detach())

            if epoch in get_knneval_epoch_from_switch(datestamp) or 0 in get_knneval_epoch_from_switch(datestamp):
                for classes_task_ind in range(len(seen_classes)):
                    task_classes = seen_classes[classes_task_ind]
                    train_feats, train_labels = extract_train_features(model, train_knn_loader,task_classes)
                    #print(train_feats.shape, train_labels.shape)
                    test_feats, test_labels   = extract_train_features(model, test_loader,task_classes)
                    #print(train_feats.shape, train_labels.shape,test_feats.shape, test_labels.shape)
                    acc = knn_eval(train_feats, train_labels, test_feats, test_labels)
                    with open(logfile, 'a') as f:
                        f.write(f"Epoch {epoch} KNN eval on Task {classes_task_ind} accuracy: {acc:.2f}%\n")
                    knn_accs.append((epoch, acc))
                    print(f"Epoch {epoch} KNN eval on Task {classes_task_ind} accuracy: {acc:.2f}%\n")
            logger.info(f"============ Starting task {task_number} epoch {epoch} ...  knn_switch_{datestamp}   ============")

            # REMOVED: train_loader.sampler.set_epoch(epoch)

            # optionally starts a queue
            if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:# and _task_number == 0:
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
                                        task_number,
                                        _task_number,
                                        epoch,
                                        lr_schedule,
                                        queue,
                                        logfile=logfile,
                                        sinkhorn_tracker = sinkhorn_tracker)
            training_stats.update(scores)
            with open(logfile, 'a') as f:
                f.write(
                    f"[SinkhornTracker][Epoch {epoch}], min_row_sum={sinkhorn_tracker.min_row_sum:.3e}, min_col_sum={sinkhorn_tracker.min_col_sum:.3e}"
                )
            logger.info(
                f"[SinkhornTracker][Epoch {epoch}] "
                f"min_row_sum={sinkhorn_tracker.min_row_sum:.3e}, "
                f"min_col_sum={sinkhorn_tracker.min_col_sum:.3e}"
            )
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
            if epoch in get_save_epoch_from_switch(datestamp) or 0 in get_save_epoch_from_switch(datestamp):
                
                print("Saving checkpoint at epoch ", epoch)
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if args.use_fp16:
                    # save_dict["amp"] = apex.amp.state_dict()
                    save_dict["scaler"] = scaler.state_dict()
                path = os.path.join('', f"checkpoint_datestamp_{datestamp}_task_{task_number}_epoch_{epoch}.pth.tar")
                torch.save(
                    save_dict,
                    path,
                )
            

            proto_hash_now = tensor_hash(model.prototypes.weight.detach())
            logger.info(f"[Proto Hash] {proto_hash_ref} -> {proto_hash_now}")

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
def tensor_hash(x):
    return hashlib.md5(x.cpu().numpy().tobytes()).hexdigest()
from torch.amp import autocast, GradScaler
def train(train_loader, model, optimizer,scaler,task,_task, epoch, lr_schedule, queue,logfile='train_log.txt',sinkhorn_tracker=None):
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
                        check_tensor("out(before_queue)", out, logger)

                        if it % 10 == 0 :
                             
                            # logit_std = out.std(dim=1).mean().item ()   
                            # logit_abs_mean = out.abs().mean().item()
                            # logger.info(
                            #     f"[Debug] logits std per sample: {logit_std:.4f}, "
                            #     f"abs mean: {logit_abs_mean:.4f}"
                            # )
                            logger.info(f"[Debug] out stats: {tensor_stats(out)}")
                            with open(logfile, 'a') as f:
                                f.write(
                                    f"[Debug] out stats: {tensor_stats(out)}\n"
                                )
                        if queue is not None:
                            if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                                use_the_queue = True
                                # out = torch.cat((
                                #     torch.mm(queue[i], model.prototypes.weight.t()),
                                #     out,
                                # ))
                                mm_part = torch.mm(queue[i], model.prototypes.weight.t())
                                check_tensor("queue[i]", queue[i], logger)
                                check_tensor("prototypes.weight", model.prototypes.weight, logger)
                                check_tensor("mm_part(queue·W^T)", mm_part, logger)
                                mm_max = mm_part.max().item()
                                mm_min = mm_part.min().item()
                                batch_max = out.max().item()
                                batch_min = out.min().item()

                                logger.info(
                                    f"[OutSource] mm_part: min={mm_min:.4f}, max={mm_max:.4f} | "
                                    f"batch_out: min={batch_min:.4f}, max={batch_max:.4f}"
                                )
                                with open(logfile, 'a') as f:
                                    f.write(
                                        f"[OutSource] mm_part: min={mm_min:.4f}, max={mm_max:.4f} | batch_out: min={batch_min:.4f}, max={batch_max:.4f}\n"
                                    ) 
                                out = torch.cat((mm_part, out))

                            nmb_crops = np.sum(args.nmb_crops)
                            embedding = embedding.view(nmb_crops, bs, -1).contiguous()
                            crop_emb = embedding[crop_id]
                            queue[i, bs:] = queue[i, :-bs].clone()
                            #queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]
                            queue[i, :bs] = crop_emb
                        
                        # check_tensor("out(before_sinkhorn)", out, logger)
                        # q = distributed_sinkhorn(out)[-bs:]
                        # check_tensor("Q(after_sinkhorn)", q, logger)

                        check_tensor("out(before_sinkhorn)", out, logger)
                        if sinkhorn_tracker != None:
                            sinkhorn_tracker.reset()

                        q = distributed_sinkhorn(out,tracker=sinkhorn_tracker,clamp_min=args.clamp_min)[-bs:]


                         


                        # --------- BEGIN: Q(after_sinkhorn) ANALYSIS CATCH ---------
                        if not torch.isfinite(q).all():
                            logger.error("========== Q(after_sinkhorn) NON-FINITE DETECTED ==========")

                            # 1) Q 的基本状态（这是结果，不是原因）
                            logger.error(
                                f"[Q] shape={tuple(q.shape)} "
                                f"finite_ratio={(torch.isfinite(q).float().mean().item()):.6f}"
                            )

                            # 2) out(before_sinkhorn) 的数值尺度（最重要的因果线索）
                            of = out[torch.isfinite(out)]
                            logger.error(
                                "[CauseCheck] out(before_sinkhorn): "
                                f"min={of.min().item():.3e}, "
                                f"max={of.max().item():.3e}, "
                                f"mean={of.mean().item():.3e}, "
                                f"std={of.std(unbiased=False).item():.3e}"
                            )

                            # 3) out / epsilon 的有效尺度（直接对应 exp 是否 underflow）
                            eps = args.epsilon
                            scaled = of / eps
                            logger.error(
                                "[CauseCheck] out/epsilon: "
                                f"min={scaled.min().item():.3e}, "
                                f"max={scaled.max().item():.3e}"
                            )

                            # 4) 判断是否为典型 underflow → sumQ = 0 型失败
                            # （即：exp(out/eps) 全为 0）
                            if scaled.max().item() < -50:
                                logger.error(
                                    "[Conclusion] Likely cause: EXP UNDERFLOW in Sinkhorn "
                                    "(out/epsilon << 0 → exp → 0 → sumQ=0 → NaN)"
                                )

                            # 5) mm_part 是否主导了 out 的尺度
                            if 'mm_part' in locals():
                                mmf = mm_part[torch.isfinite(mm_part)]
                                logger.error(
                                    "[CauseCheck] mm_part(queue·W^T): "
                                    f"min={mmf.min().item():.3e}, "
                                    f"max={mmf.max().item():.3e}, "
                                    f"std={mmf.std(unbiased=False).item():.3e}"
                                )

                                if mmf.abs().max().item() > 20:
                                    logger.error(
                                        "[Conclusion] mm_part scale dominates out → "
                                        "Sinkhorn input distribution skewed by queue/prototypes"
                                    )

                            # 6) queue 与 prototype 的范数状态（是否发生尺度漂移）
                            if queue is not None:
                                qi = queue[i]
                                logger.error(
                                    "[CauseCheck] queue[i] norm: "
                                    f"mean={qi.norm(dim=1).mean().item():.3e}, "
                                    f"max={qi.norm(dim=1).max().item():.3e}"
                                )

                            w = model.prototypes.weight
                            logger.error(
                                "[CauseCheck] prototypes.weight norm: "
                                f"mean={w.norm(dim=1).mean().item():.3e}, "
                                f"max={w.norm(dim=1).max().item():.3e}"
                            )

                            # 7) 给出“结构性原因”的判定（不是修复）
                            logger.error(
                                "[Final Diagnosis] Q becomes all-NaN *after* Sinkhorn, "
                                "while inputs remain finite ⇒ "
                                "this is a Sinkhorn normalization degeneracy, "
                                "not a forward NaN propagation."
                            )

                            logger.error("============================================================")
                            raise RuntimeError("Non-finite Q(after_sinkhorn) — analysis complete")

                        # --------- END: Q(after_sinkhorn) ANALYSIS CATCH ---------




                        with torch.no_grad():
                            all_Q_epoch.append(q.detach().cpu())
                    if it % 10 == 0:
                        #print q.std, q.mean()
                        if True:
                            m = compute_batch_metrics(q)
                            logger.info(
                                f"[Metric] Epoch {epoch} Iter {it} "
                                    f"H(Q)={m['q_entropy']:.5f} "
                                    f"maxQ={m['q_maxprob']:.5f} gap={m['q_top1_top2_gap']:.5f} "
                                    f"used_proto={m['q_num_used_proto']:.5f}\n"
                            )
                            with open(logfile, "a") as f:
                                f.write(
                                    f"[Q-AfterSinkhorn] Epoch {epoch} Iter {it} "
                                    f"q_entropy={m['q_entropy']:.5f} "
                                    f"q_maxprob={m['q_maxprob']:.5f} q_top1_top2_gap={m['q_top1_top2_gap']:.5f} "
                                    f"q_num_used_proto={m['q_num_used_proto']:.5f}\n"
                                )
                        logger.info(f"Sinkhorn Q: std {q.std().item():.4f}, mean {q.mean().item():.4f}")
                        with open(logfile, 'a') as f:
                            f.write(f"Task {task} Epoch {epoch} Iter {it} Sinkhorn Q: std {q.std().item():.4f}, mean {q.mean().item():.4f}\n")   

                    subloss = 0
                    for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                        x = output[bs * v: bs * (v + 1)] / args.temperature
                        check_tensor("x(before_log_softmax)", x, logger)
                        check_tensor("log_softmax(x)", F.log_softmax(x, dim=1), logger)
                        check_tensor("q * log_softmax(x)", q * F.log_softmax(x, dim=1), logger)
                        subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
                    loss += subloss / (np.sum(args.nmb_crops) - 1)
                loss /= len(args.crops_for_assign)
                if not torch.isfinite(loss):
                    logger.info("Loss is NaN/Inf — dumping diagnostics and stopping.")
                    # dump out_all min/max, scaled.max, mm_part stats, queue stats, etc.
                    torch.save({
                        "epoch": epoch,
                        "iter": it,
                        "task": task,
                        "out": out.detach().cpu(),
                        "q": q.detach().cpu(),
                        "queue_i": queue[i].detach().cpu() if queue is not None else None,
                        "prototypes": model.prototypes.weight.detach().cpu(),
                    }, f"nan_dump_task{task}_epoch{epoch}_iter{it}.pt")
                    raise RuntimeError("NaN loss")
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
            if iteration < args.freeze_prototypes_niters:# and _task ==0:
                for name, p in model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if iteration < args.freeze_prototypes_niters:# and _task ==0:
                for name, p in model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
            optimizer.step()
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
         
        if args.rank ==0 and it % 10 == 0:
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
def tensor_stats(t):
    return (
        f"min={t.min().item():.3e}, "
        f"max={t.max().item():.3e}, "
        f"mean={t.mean().item():.3e}, "
        f"std={t.std().item():.3e}, "
        f"l2={t.norm(dim=1).mean().item():.3e}"
    )
def check_tensor(name, t, logger, max_print=5):
    if t is None:
        return

    finite_mask = torch.isfinite(t)
    if finite_mask.all():
        return

    # 防止全 NaN / 全 Inf
    if finite_mask.any():
        t_finite = t[finite_mask]
        t_min = t_finite.min().item()
        t_max = t_finite.max().item()
        t_mean = t_finite.mean().item()
        t_std = t_finite.std().item()
    else:
        t_min = t_max = t_mean = t_std = float("nan")

    bad = ~finite_mask

    logger.error(
        f"[NaN DETECTED] {name}\n"
        f"  shape={tuple(t.shape)} dtype={t.dtype} device={t.device}\n"
        f"  finite_ratio={finite_mask.float().mean().item():.4f}\n"
        f"  min={t_min:.4e}, max={t_max:.4e}, "
        f"mean={t_mean:.4e}, std={t_std:.4e}\n"
        f"  bad_count={bad.sum().item()}"
    )

    # 打印前几个 NaN / Inf 的 index
    idx = bad.nonzero(as_tuple=False)[:max_print]
    logger.error(f"  bad_indices(sample)={idx.tolist()}")

    raise RuntimeError(f"Non-finite tensor detected: {name}")

 
def compute_batch_metrics(q: torch.Tensor, eps: float = 1e-12) -> dict:
    """
    q: [B, K] Sinkhorn assignment
    returns: python floats
    """
    q_ = q.clamp(min=eps)
    entropy = -(q_ * q_.log()).sum(dim=1).mean()  # mean over batch

    top2 = q_.topk(2, dim=1).values              # [B, 2]
    max_prob = top2[:, 0].mean()
    top1_top2_gap = (top2[:, 0] - top2[:, 1]).mean()

    assigned = q_.argmax(dim=1)
    num_used_proto = assigned.unique().numel()

    return {
        "q_entropy": float(entropy.item()),
        "q_maxprob": float(max_prob.item()),
        "q_top1_top2_gap": float(top1_top2_gap.item()),
        "q_num_used_proto": float(num_used_proto),
    }
@torch.no_grad()
def distributed_sinkhorn(out, tracker=None,clamp_min=None):
    if True: 
        out = out.float() # per-sample stabilization: ensure max exponent is 0 
        out = out / args.epsilon 
        out = out - out.max(dim=1, keepdim=True).values # <= 0
        if clamp_min is not None:
            out = out.clamp(min=clamp_min)
        Q = torch.exp(out).t()  # now exp in (0, 1]
        Q = Q / (Q.sum() + 1e-12)
        B = Q.shape[1]
        K = Q.shape[0]
    else:
        Q = torch.exp(out / args.epsilon).t()   # K x B
        B = Q.shape[1]
        K = Q.shape[0]

        # make the matrix sum to 1
        Q /= torch.sum(Q)
         
     

    for _ in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        Q /= (torch.sum(Q, dim=1, keepdim=True) + 1e-12)
        Q /= K
        if tracker is not None:
            tracker.update(Q)
        # normalize each column: total weight per sample must be 1/B
        Q /= (torch.sum(Q, dim=0, keepdim=True) + 1e-12)
        Q /= B
        if tracker is not None:
            tracker.update(Q)

    Q *= B  # columns sum to 1
    return Q.t()  # B x K
 


if __name__ == "__main__":
    main()
