import argparse
import math
import os
from random import random
import shutil
import time
from logging import getLogger
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from hessian_energy_tracker import HessianEnergyTrackerSwAV
from curvature_skip_controller import CurvatureSkipController
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
parser.add_argument("--epsilon", default=0.01, type=float)
parser.add_argument("--l2_reg", default=1, type=float)
parser.add_argument("--sinkhorn_iterations", default=10, type=int)
parser.add_argument("--sinkhorn_method", type=str, default="swav_native", choices=["swav_native", "entropic_POT", "L2_regularized_POT"] )
parser.add_argument("--feat_dim", default=128, type=int)
parser.add_argument("--nmb_prototypes", default=250, type=int)
parser.add_argument("--queue_length", type=int, default=0)
parser.add_argument("--epoch_queue_starts", type=int, default=5)
parser.add_argument("--queue_cleared_each_task", type=bool_flag, default=False)

#########################
#### optim parameters ###
#########################
parser.add_argument("--dataset", default="cifar100", type=str, choices=["cifar100","tinyimagenet","food101"])    
parser.add_argument("--epochs", default=120, type=int)
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
parser.add_argument("--seed", type=int, default=47)
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
            CICL=args.CICL,
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
    elif args.dataset == "food101":
        from src.multicropfood101 import MultiCropDataset
        base_dataset = MultiCropDataset(
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            # CICL=args.CICL,
            train=True,
            val = True
        )
         
    elif args.dataset == "cifar100":
        probe_transform = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
    ])
        base_dataset = datasets.CIFAR100(
            root='./',
            train=True,
            transform=probe_transform,
            download=True,
        )
         
    if args.dataset == 'cifar100':
        num_samples = min(256, len(base_dataset))
    else:
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
def compute_probe_kl(model, probe_loader, args, prev_Q=None, device="cuda",epoch = -1):
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

            Q, Q_joint = distributed_sinkhorn(logits,epoch =epoch,total_epoch=args.epochs)        # 你已在 sinkhorn 内 out.float() 更好；这里也确保 logits fp32

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


@torch.no_grad()
def eval_swav_loss_on_loader(model, loader, args, device="cuda", max_batches=20, epoch=-1):
    '''
    Evaluate SwAV loss (no grad) on a given loader (e.g., previous-task data).
    Uses the same distributed_sinkhorn + swav_loss_from_q as training.
    '''
    model.eval()
    losses = []
    with torch.no_grad():
    #if True:
        for bi, inputs in enumerate(loader):
            if bi >= max_batches:
                break
            
            _, out = model(inputs)
            bs = inputs[0].size(0)

            q_assign = {}
            out_det = out.detach().float()
            for crop_id in args.crops_for_assign:
                logits = out_det[bs * crop_id: bs * (crop_id + 1)]
                q, _ = distributed_sinkhorn(
                    logits,
                    tracker=None,
                    clamp_min=args.clamp_min,
                    epoch=epoch,
                    total_epoch=args.epochs,
                )
                q_assign[crop_id] = q[-bs:].contiguous()

            loss = model.swav_loss_from_q(
                output=out.float(),
                q_assign=q_assign,
                bs=bs,
                temperature=args.temperature,
                nmb_crops=int(np.sum(args.nmb_crops)),
                crops_for_assign=args.crops_for_assign,
            )
            losses.append(float(loss.item()))
    model.train()
    if len(losses) == 0:
        return float("nan")
    return float(sum(losses) / len(losses))


def main():
    global args
    args = parser.parse_args()
    with open("knn_switch_"+datestamp, 'w') as f:
        #f.write("epoch: [1,30,60,90,99,120,150,180,210,240,270,300,330,360,390,399]\n")
        f.write("epoch: [1,10,30,60,90,119]\n")
        f.write(args.__repr__()+"\n")
        f.write("save_epoch: [119]\n")
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
        val_dataset = MultiCropDataset(
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            CICL=CICL,
            train=False
        )
    if args.dataset == "food101":
        from src.multicropfood101 import MultiCropDataset, TaskSubset
        train_dataset = MultiCropDataset(
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            CICL=CICL,
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
    val_loaders = []
    train_loader = None
    val_loader = None
    task_class_lists = []
     
    if CICL:
        if args.dataset == 'cifar100':
            full_labels = np.array(train_dataset.dataset.targets)  # shape=(50000,)
            full_labels_val = np.array(val_dataset.dataset.targets)  # shape=(10000,)
        elif args.dataset == 'tinyimagenet':
            full_labels =  train_dataset.full_selected_labels()  # shape=(100000,)
        elif args.dataset == 'food101':
            # full_labels = np.array(train_dataset.dataset['train']['label'])  # shape=(75750,)
            # full_labels = full_labels[full_labels < 100] 
            full_labels = np.array(train_dataset.dataset['label'][train_dataset.indices])
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
        task_indices_list_for_val = []
        for class_list in task_class_lists:
            mask = np.isin(full_labels, class_list)
            indices = np.where(mask)[0]
            task_indices_list.append(indices)
            mask_val = np.isin(full_labels_val, class_list)
            indices_val = np.where(mask_val)[0]
            task_indices_list_for_val.append(indices_val)
         

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
        for t,indices in enumerate(task_indices_list_for_val):
            subset = TaskSubset(val_dataset, indices)
            loader = torch.utils.data.DataLoader(
                subset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
            )
            # for _, input in enumerate(loader):
            #     a = len(input)
            #     pass
            val_loaders.append(loader)
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
    if args.dataset == "food101":
        from src.multicropfood101 import MultiCropDataset
        test_dataset = MultiCropDataset(
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            CICL=CICL,
            train=False
        )
        train_knn_dataset = MultiCropDataset(
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
            CICL=CICL,
            train=True,
            val = True
        )
        test_dataset = SubsampledDataset(test_dataset, ratio=0.5, seed=0)
        #train_knn_dataset = SubsampledDataset(train_knn_dataset, ratio=0.1, seed=0) 
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
        test_dataset = SubsampledDataset(test_dataset, ratio=0.5, seed=0)
        train_knn_dataset = SubsampledDataset(train_knn_dataset, ratio=0.2, seed=0) 
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
    os.mkdir("log_"+datestamp)
    os.mkdir(f"log_{datestamp}/cost_heatmap")
    os.mkdir(f"log_{datestamp}/hessian_energy_swav")
    logfile = f"log_{datestamp}/train_log.txt"

    with open(logfile, 'a') as f:
        f.write(args.__repr__()+"\n")
    model = model.cuda()
    logger.info(model)
    logger.info("Building model done.")
    seen_classes = []
    knn_accs = []
    cost_probe_loaders = []
    save_root = os.path.join(f"log_{datestamp}", f"hessian_energy_swav")
    #save_root = os.path.join(save_root, f"hessian_energy_swav_task{task_number}")
    tracker = HessianEnergyTrackerSwAV(
                anchor_epochs=[],#[anchor for anchor in range(args.epochs)],
                window=3,
                top_k_theta=80,       
                top_k_C=800,         
                bin_size=50,
                save_root=save_root,
                device="cuda",
                crops_for_assign=args.crops_for_assign,
                nmb_crops=int(np.sum(args.nmb_crops)),
                temperature=args.temperature,
                clamp_min=args.clamp_min,
                total_epoch=args.epochs,
                sinkhorn_fn=distributed_sinkhorn,  # 关键：用你训练里的那一个
            )    
    
     
    skip_controller = CurvatureSkipController(
        skip_mode="hessian",     # or "random"
        tau_curr_c=-1,            # -1 disable; >=0 enable
        tau_prev_c=-1,            # -1 disable; >=0 enable
        tau_curr_theta=-1,            # -1 disable; >=0 enable
        tau_prev_theta=-1,            # -1 disable; >=0 enable
        m_anchor_C=-1,
        m_anchor_theta=-1,
        m_optimal_C=-1,
        m_optimal_theta=-1,
        topk_C=tracker.top_k_C,  # 你也可以自己设一个固定值，或者直接用 tracker 的那个
        topk_theta=tracker.top_k_theta,  # 你也可以自己设一个固定值，或者直接用 tracker 的那个
        use_theta=False,
        use_C=False,
        replay_skip_log=None,
        device="cuda",
    )
    skip_controller = None  # disable for now
    with open(logfile, 'a') as f:
        f.write(f"Skip controller config: {skip_controller.__dict__ if skip_controller is not None else 'None'}\n") 
        f.write(f'tracker config: {tracker.__dict__}\n')
    for _task_number in range(min(2, len(train_loaders))):
        task_number = _task_number
        if task_number == 1:
            tracker.anchor_epochs = [anchor for anchor in range(args.epochs)]
        train_loader = train_loaders[task_number]
        if _task_number > 0:
            previous_train_loader = train_loaders[task_number-1]
        prev_eval_loader = train_loaders[task_number-1] if task_number >0 else None   
        
        cost_probe_loader =  build_cost_probe_batch(train_loader.dataset, args.batch_size, device="cuda")
        cost_probe_loaders.append(cost_probe_loader)
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


        if True:
            
                
            probe_batches = []
            probe_batches_previous_task = []
            for i, (inputs, _) in enumerate(train_loader):
                print(type(inputs), len(inputs), inputs[0].shape, inputs[1].shape)
                inputs_small = inputs[:32]
                print(inputs_small[0].shape, inputs_small[1].shape)
                probe_batches.append(inputs_small)
                if i == 0:   
                    break
            if _task_number > 0:   
                for i, (inputs, _) in enumerate(previous_train_loader):
                    inputs_small = inputs[:32]
                    probe_batches_previous_task.append(inputs_small)
                    if i == 0:   
                        break
            else:
                    probe_batches_previous_task = None
            tracker.start_task(model, probe_batches, probe_batches_previous_task, epoch0=0,task=task_number)
            if skip_controller is not None:
                skip_controller.start_task(task=task_number, tracker=tracker)
        for epoch in range(0, args.epochs):
             
            proto_hash_ref = tensor_hash(model.prototypes.weight.detach())
            if prev_eval_loader is not None:
                prev_loss = eval_swav_loss_on_loader(model, prev_eval_loader, args, device="cuda", max_batches=20, epoch=epoch)
                with open(logfile, 'a') as f:
                    f.write(f"[PrevTaskLoss] BEFORE task={task_number} epoch={epoch} prev_task=0 swav_loss={prev_loss:.6f}\n")
                logger.info(f"[PrevTaskLoss] BEFORE task={task_number} epoch={epoch} prev_task=0 swav_loss={prev_loss:.6f}")

            if epoch in get_knneval_epoch_from_switch(datestamp) or 0 in get_knneval_epoch_from_switch(datestamp):
                for cost_probe_loader_idx in range(len(cost_probe_loaders)):
                    _cost_probe_loader = cost_probe_loaders[cost_probe_loader_idx]
                    
                    result = plot_cost_matrix(_cost_probe_loader[0],model,args,path=f"log_{datestamp}/cost_heatmap/cost_data{cost_probe_loader_idx}_onTask{task_number}_epoch{epoch}.png", task=task_number, epoch=epoch, data_index=cost_probe_loader_idx)
                    print(f"Epoch {epoch} Cost matrix plot for data from task {cost_probe_loader_idx} on model of task {task_number} saved to log_{datestamp}/cost_heatmap/cost_data{cost_probe_loader_idx}_onTask{task_number}_epoch{epoch}.png    ")
                     

                     

                 
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
            if skip_controller is not None:
                skip_controller.start_epoch(task=task_number, epoch=epoch, total_steps=len(train_loader),tracker=tracker)

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
                                        sinkhorn_tracker = sinkhorn_tracker,hessian_tracker = tracker, skip_controller=skip_controller)
            training_stats.update(scores)
            tracker.after_epoch(epoch,model,task_number)
            if skip_controller is not None:
                skip_controller.end_epoch()
            # if epoch == args.epochs - 1:
            #      tracker.end_task()

            if epoch == args.epochs - 1:
                 
                prev_loss = eval_swav_loss_on_loader(model, train_loader, args, device="cuda", max_batches=20, epoch=epoch)
                with open(logfile, 'a') as f:
                    f.write(f"[PrevTaskLoss] AFTER task={task_number} epoch={epoch} prev_task=0 swav_loss={prev_loss:.6f}\n")
                logger.info(f"[PrevTaskLoss] AFTER task={task_number} epoch={epoch} prev_task=0 swav_loss={prev_loss:.6f}")

            #if epoch >= 40:
            #    print(torch.cuda.memory_summary())
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
                epoch = epoch
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
                    save_dict["scaler"] = scaler.state_dict()
                path = os.path.join('', f"checkpoint_datestamp_{datestamp}_task_{task_number}_epoch_{epoch}.pth.tar")
                torch.save(
                    save_dict,
                    path,
                )
            

            proto_hash_now = tensor_hash(model.prototypes.weight.detach())
            logger.info(f"[Proto Hash] {proto_hash_ref} -> {proto_hash_now}")

             
def tensor_hash(x):
    return hashlib.md5(x.cpu().numpy().tobytes()).hexdigest()
from torch.amp import autocast, GradScaler
q_assign_test = {}
def train(train_loader, model, optimizer,scaler,task,_task, epoch, lr_schedule, queue,logfile='train_log.txt',sinkhorn_tracker=None,hessian_tracker = None, skip_controller=None):
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
                        
                         
                        check_tensor("out(before_sinkhorn)", out, logger)
                        if sinkhorn_tracker != None:
                            sinkhorn_tracker.reset()

                        q,q_joint = distributed_sinkhorn(out,
                        tracker=sinkhorn_tracker,
                        clamp_min=args.clamp_min,
                        epoch = epoch,
                        total_epoch = args.epochs)[-bs:]
                        q_assign_test[crop_id] = q
                         


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
                            m_joint = compute_joint_metrics(q_joint)
                            logger.info(
                                f"[Metric] Epoch {epoch} Iter {it} "
                                    f"H(Q_joint)={m_joint['joint_col_entropy']:.5f} "
                                    f"minQ_joint={m_joint['joint_col_min']:.5f} maxQ_joint={m_joint['joint_col_max']:.5f} "
                                    f"GiniQ_joint={m_joint['joint_col_gini']:.5f} used_proto_joint={m_joint['num_active_proto_joint']:.5f}\n"
                            )
                            with open(logfile, "a") as f:
                                f.write(
                                    f"[Q_joint-AfterSinkhorn] Epoch {epoch} Iter {it} "
                                    f"joint_col_entropy={m_joint['joint_col_entropy']:.5f} "
                                    f"joint_col_min={m_joint['joint_col_min']:.5f} joint_col_max={m_joint['joint_col_max']:.5f} "
                                    f"joint_col_gini={m_joint['joint_col_gini']:.5f} joint_num_active_proto={m_joint['num_active_proto_joint']:.5f}\n"
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
             assert False, "FP16 training is currently required for Sinkhorn stability. Remove this assertion if you want to run in FP32 (not recommended)."

        
        optimizer.zero_grad()
        if args.use_fp16:
            scaler.scale(loss).backward()
            if iteration < args.freeze_prototypes_niters:# and _task ==0:
                for name, p in model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None

            # -----------------------------
            # Curvature-aware step skipping (after backward, before optimizer step)
            # Rule: ratio = ||U^T g||^2 / ||g||^2  (g: current gradient for chosen block)
            if skip_controller is not None:
                skip_blocks, skip_stats = skip_controller.should_skip(
                    task=task,
                    epoch=epoch,
                    step=it,
                    model=model,
                )
                skip_stats["skip_decision"] = False  # default to no skip
                if len(skip_blocks) > 0:
                    if True:
                        anchor_ratio = skip_stats["theta/optimal_ratio"]
                        anchor_tau = skip_stats["theta/optimal_tau"]
                        eps = 1e-12
                        tau = max(anchor_tau, eps)

                        r = max(0.0, (anchor_ratio - tau) / tau)   # 相对超出比例
                        p_max = 0.7

                        k = 2.0  # 斜率/激进程度（下面给你如何定）
                        p_skip = p_max * (1.0 - math.exp(-k * r))

                        skip = np.random.rand() < p_skip


                        skip_stats["skip_decision"] = skip   
                    # ---- block-wise gradient masking ----
                    if "C" in skip_blocks and skip:
                        for name, p in model.named_parameters():
                            if "prototypes" in name:
                                p.grad = None

                    if "theta" in skip_blocks and skip:
                        for name, p in model.named_parameters():
                            if "prototypes" not in name:
                                p.grad = None

                    # logging (optional)
                    if skip_blocks:
                        with open(logfile, "a") as f:
                            f.write(
                                f"[CurvSkip] task={task} epoch={epoch} iter={it} "
                                f"skip_blocks={sorted(skip_blocks)}\n"
                                f"skip_stats={skip_stats}\n"
                            )               
                print(f"[CurvSkip]   iter={it} {skip_stats}")
            # ---- ALWAYS step optimizer & scaler ----
            scaler.step(optimizer)
            scaler.update()

            # ---- after_step should be AFTER optimizer.step ----
            hessian_tracker.after_step(task, epoch, it, model)


        else:
            assert False, "FP16 training is currently required for Sinkhorn stability. Remove this assertion if you want to run in FP32 (not recommended)."
        losses.update(loss.item(), inputs[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
         

        if it % 10 == 0:
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
def debug_conditions(out, Q_local, Q_pot, eps):
    # Q_*: [B,K] with rows summing to 1 (after your final Q*=B and transpose)

    diff = Q_local - Q_pot
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    rel_max = (diff.abs() / (Q_pot.abs() + 1e-12)).max().item()

    # marginals
    B, K = Q_local.shape
    row_err_local = (Q_local.sum(dim=1) - 1.0).abs().max().item()
    col_target = B / K
    col_err_local = (Q_local.sum(dim=0) - col_target).abs().max().item()

    row_err_pot = (Q_pot.sum(dim=1) - 1.0).abs().max().item()
    col_err_pot = (Q_pot.sum(dim=0) - col_target).abs().max().item()

    # objective (normalize to K×B, sum=1 form)
    C = (-out).T  # [K,B]
    def ot_obj(Q_BK):
        Q = (Q_BK.T / B).clamp_min(1e-30)  # [K,B], sum=1
        return (Q * C).sum() + eps * (Q * (Q.log() - 1.0)).sum()

    obj_local = ot_obj(Q_local).item()
    obj_pot   = ot_obj(Q_pot).item()
    obj_rel = abs(obj_local - obj_pot) / (abs(obj_pot) + 1e-12)

    should_debug = (
        (max_abs > 1e-3) or
        (mean_abs > 1e-5) or
        (rel_max > 1e-2) or
        (row_err_local > 1e-3) or
        (col_err_local > 1e-3) or
        (obj_rel > 1e-3)
    )
    if should_debug == False:
        print(f"max|ΔQ|={max_abs:.3e}, mean|ΔQ|={mean_abs:.3e}, rel_max={rel_max:.3e}")
        print(f"row_err local={row_err_local:.3e}, pot={row_err_pot:.3e}")
        print(f"col_err local={col_err_local:.3e}, pot={col_err_pot:.3e} (target={col_target:.6f})")
        print(f"obj local={obj_local:.6e}, pot={obj_pot:.6e}, rel={obj_rel:.3e}")
 
 
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

def compute_joint_metrics(q_joint: torch.Tensor, eps=1e-12):
    qj = q_joint.clamp(min=eps)
    proto_mass = qj.sum(dim=0)  # K

    return {
        "joint_col_entropy": float(-(proto_mass * proto_mass.log()).sum().item()),
        "joint_col_min": float(proto_mass.min().item()),
        "joint_col_max": float(proto_mass.max().item()),
        "joint_col_gini": float(
            1 - (proto_mass / proto_mass.sum()).pow(2).sum().item()
        ),
        "num_active_proto_joint": int((proto_mass > 1e-6).sum().item()),
    }

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
 
 
import ot
@torch.no_grad()
def pot_sinkhorn_Q(out, epsilon):
    """
    out: torch.Tensor, shape [B, K]
    returns: torch.Tensor, shape [B, K]
    """

    B, K = out.shape

    # cost matrix: K × B
    C = (-out).T.detach().cpu().numpy()

    # uniform marginals
    a = np.ones(K) / K      # prototype marginal
    b = np.ones(B) / B      # sample marginal

    # entropic OT
    Q = ot.sinkhorn(
        a, b, C,
        reg=epsilon,
        numItermax=1000,
        stopThr=1e-9,
        verbose=False
    )

    # POT returns K × B
    Q = torch.from_numpy(Q).to(out.device, out.dtype)

    # match your convention: columns sum to 1
    Q = Q * B

    return Q.T   # B × K

@torch.no_grad()
def computing_Q_entropic_SwAV_native_implementation(out,  tracker=None,clamp_min=None):
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
    Q_joint = Q.clone()
    Q *= B  # columns sum to 1
    return Q.t(), Q_joint.t()  # B x K
@torch.no_grad()
def computing_Q_entropic_POT_implementation(out, tracker=None, clamp_min=None):
    # ---- keep your preprocessing ----
    out = out.float()
    out = out / args.epsilon
    out = out - out.max(dim=1, keepdim=True).values
    if clamp_min is not None:
        out = out.clamp(min=clamp_min)

    B, K = out.shape

    # ---- POT cost & marginals ----
    C = (-out).T.detach().cpu().numpy()   # K × B
    a = np.ones(K) / K
    b = np.ones(B) / B

    # ---- POT Sinkhorn (entropic OT) ----
    Q = ot.sinkhorn(
        a, b, C,
        reg=1.0,                    # already absorbed by out/epsilon
        numItermax=args.sinkhorn_iterations,
        stopThr=1e-9,
        verbose=False
    )

    # ---- back to torch & match convention ----
    Q = torch.from_numpy(Q).to(out.device, out.dtype)
    Q = Q * B                       # columns sum to 1
    Q = Q.T                         # B × K

    if tracker is not None:
        tracker.update(Q)
    #check if there is negtaive elements in Q
    if (Q < 0).any():
        logger.error("========== Q(after_sinkhorn) NEGATIVE DETECTED ==========")
        logger.error(
            f"[Q] shape={tuple(Q.shape)} "
            f"negative_ratio={(Q < 0).float().mean().item():.4f} "
            f"min={Q.min().item():.4e}, max={Q.max().item():.4e}"
        )
        raise RuntimeError("Negative Q(after_sinkhorn) — analysis complete")

    return Q


def distributed_sinkhorn(out, tracker=None, clamp_min=None,epoch=-1,total_epoch=100):
    if args.sinkhorn_method == "swav_native":
        return computing_Q_entropic_SwAV_native_implementation(
            out,
            tracker=tracker,
            clamp_min=clamp_min,
        )
    elif args.sinkhorn_method == "entropic_POT":
        return computing_Q_entropic_POT_implementation(
            out,
            tracker=tracker,
            clamp_min=clamp_min,
        )
    elif args.sinkhorn_method == "L2_regularized_POT":
        return computing_Q_L2_regularized_POT_implementation(
            out,
            tracker=tracker,
            clamp_min=clamp_min,
            epoch=epoch,
            total_epoch=total_epoch
        )
    else:
        raise ValueError(f"Unknown sinkhorn_method: {args.sinkhorn_method}")

def computing_Q_L2_regularized_POT_implementation(out, tracker=None, clamp_min=None,epoch=-1,total_epoch=100):
     

    def l2_lambda(epoch, total_epoch, l2_reg):
        t = epoch / max(total_epoch - 1, 1)
        return l2_reg * 0.5 * (1 - math.cos(math.pi * t))
    out = out.float()
    #out = out / args.epsilon
    out = out - out.max(dim=1, keepdim=True).values
    if clamp_min is not None:
        out = out.clamp(min=clamp_min)

    B, K = out.shape

    # ---- POT cost & marginals (unchanged) ----
    C = (-out).T.detach().cpu().numpy()   # K × B
    a = np.ones(K, dtype=np.float64) / K
    b = np.ones(B, dtype=np.float64) / B
    assert epoch >= 0, "L2 POT requires explicit epoch for lambda scheduling"

    if epoch == -1:
        cur_lambda = args.l2_reg
    else:
        cur_lambda = l2_lambda(
            epoch=epoch,
            total_epoch=total_epoch,
            l2_reg=args.l2_reg
        )

    # ---- POT quadratic/L2-regularized OT ----
    # POT: res = ot.solve(M, a, b, reg=..., reg_type='L2') and coupling is res.plan
    # See ot.solve docs for reg_type='L2' quadratic regularized OT. :contentReference[oaicite:1]{index=1}
    res = ot.solve(
        C, a, b,
        reg=float(cur_lambda),          # <-- your lambda for L2 regularization on Q
        reg_type="L2",
        max_iter=int(args.sinkhorn_iterations),
        tol=1e-9,
        verbose=False,
    )
    Q = res.plan  # K × B

    # ---- back to torch & match SwAV convention (unchanged) ----
    Q = torch.from_numpy(Q).to(out.device, out.dtype)
    Q_joint = Q.clone()
    Q = Q * B          # so that (after transpose) rows sum to 1
    Q = Q.T            # B × K

    # ---- safety: eliminate tiny numerical negatives (should be ~0 anyway) ----
    # (ot.solve enforces T >= 0 in the formulation, but this guards float noise) :contentReference[oaicite:2]{index=2}
    Q = Q.clamp_min_(0.0)
    Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12)  # keep row-stochastic for SwAV loss

    if tracker is not None:
        tracker.update(Q)

    return Q, Q_joint.t()  # B x K
def build_cost_probe_batch(dataset, batch_size, device):
    indices = torch.randperm(len(dataset))[:batch_size]
    images = torch.stack([dataset[i][0] for i in indices])
    return [images.to(device) , indices.cpu().numpy()]

 
 

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Dict

@torch.no_grad()
def _plot_cost_matrix(
    batch_feats: torch.Tensor,               # [B, D]
    proto_weights: torch.Tensor,             # [K, D]
    *,
    Q: torch.Tensor,                         # [B, K]
    logits: Optional[torch.Tensor] = None,   # [B, K]
    cost_mode: str = "neg_dot",              # "neg_dot" | "sqeuclidean"
    normalize: bool = True,
    sort_rows: str = "argmin",
    sort_cols: str = "mass",
    title: Optional[str] = None,
    save_path: str = "./cost_and_q.png",
    dpi: int = 160,
    C_vmax_percentile: Optional[float] = 99.0,
    C_vmin_percentile: Optional[float] = 1.0,
    Q_vmax_percentile: Optional[float] = 99.5,
    Q_vmin_percentile: Optional[float] = 0.0,
    show: bool = False,

    # ---- NEW ----
    save_csv_dir: Optional[str] = None, 
    task = 0,
    epoch = 0,      
    csv_prefix: str = "debug",               # file prefix
    force_geometric_C: bool = True,          # compute C_geom for sanity even if logits is provided
    sanity_print: bool = True,               # print sanity stats
) -> Dict[str, torch.Tensor]:

    assert batch_feats.dim() == 2
    assert proto_weights.dim() == 2
    assert Q is not None and Q.dim() == 2

    device = batch_feats.device
    B, D = batch_feats.shape
    K, D2 = proto_weights.shape
    assert D == D2
    assert Q.shape == (B, K)

    # --------------------------
    # 1) Build C_used (what you plot)
    # --------------------------
    C_used = None
    if logits is not None:
        C_used = -logits
    else:
        C_used = None  # will be set to geom below

    # --------------------------
    # 1b) Build C_geom (pure geometry) for sanity
    # --------------------------
    # This lets us detect if -logits is “flattening” the visualization.
    z = batch_feats
    c = proto_weights
    if normalize and cost_mode == "neg_dot":
        z = torch.nn.functional.normalize(z, dim=1)
        c = torch.nn.functional.normalize(c, dim=1)

    if cost_mode == "neg_dot":
        C_geom = -(z @ c.t())  # [B,K]
    elif cost_mode == "sqeuclidean":
        z2 = (batch_feats ** 2).sum(dim=1, keepdim=True)
        c2 = (proto_weights ** 2).sum(dim=1, keepdim=True).t()
        sim = batch_feats @ proto_weights.t()
        C_geom = z2 + c2 - 2.0 * sim
    else:
        raise ValueError(f"Unknown cost_mode: {cost_mode}")

    if C_used is None or force_geometric_C:
        # If no logits provided, or if you prefer to plot geom anyway.
        # If logits exists and you still want to plot -logits, set force_geometric_C=False.
        if C_used is None:
            C_used = C_geom
        # else keep C_used = -logits for plotting, but still have C_geom to compare

    # to numpy
    C_used_cpu = C_used.detach().float().cpu().numpy()
    C_geom_cpu = C_geom.detach().float().cpu().numpy()
    Q_cpu = Q.detach().float().cpu().numpy()

    # --------------------------
    # 2) Row sorting (from C_used)
    # --------------------------
    row_idx = np.arange(B)
    if sort_rows == "argmin":
        row_best = np.argmin(C_used_cpu, axis=1)
        row_minc = C_used_cpu[np.arange(B), row_best]
        row_idx = np.lexsort((row_minc, row_best))
    elif sort_rows == "mincost":
        row_idx = np.argsort(np.min(C_used_cpu, axis=1))
    elif sort_rows == "none":
        pass
    else:
        raise ValueError(f"Unknown sort_rows: {sort_rows}")

    C_used_cpu = C_used_cpu[row_idx, :]
    C_geom_cpu = C_geom_cpu[row_idx, :]
    Q_cpu      = Q_cpu[row_idx, :]

    # --------------------------
    # 3) Column sorting (from Q after row-sort)
    # --------------------------
    col_idx = np.arange(K)
    if sort_cols == "mass":
        qmass = Q_cpu.sum(axis=0)
        col_idx = np.argsort(-qmass)
    elif sort_cols == "usage":
        hard = np.argmax(Q_cpu, axis=1)
        usage = np.bincount(hard, minlength=K).astype(np.float32)
        col_idx = np.argsort(-usage)
    elif sort_cols == "avgcost":
        col_avg = C_used_cpu.mean(axis=0)
        col_idx = np.argsort(col_avg)
    elif sort_cols == "none":
        pass
    else:
        raise ValueError(f"Unknown sort_cols: {sort_cols}")

    C_used_cpu = C_used_cpu[:, col_idx]
    C_geom_cpu = C_geom_cpu[:, col_idx]
    Q_cpu      = Q_cpu[:, col_idx]

    # --------------------------
    # 4) Robust scaling
    # --------------------------
    def _robust_vmin_vmax(mat: np.ndarray, vmin_p, vmax_p):
        if vmin_p is None or vmax_p is None:
            return None, None
        flat = mat.reshape(-1)
        vmin = float(np.percentile(flat, vmin_p))
        vmax = float(np.percentile(flat, vmax_p))
        # Detect “near-constant” cases too
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin >= vmax) or (abs(vmax - vmin) < 1e-6):
            return None, None
        return vmin, vmax

    C_vmin, C_vmax = _robust_vmin_vmax(C_used_cpu, C_vmin_percentile, C_vmax_percentile)

    if Q_vmin_percentile is None or Q_vmax_percentile is None:
        Q_vmin, Q_vmax = None, None
    else:
        Q_vmin = float(np.percentile(Q_cpu.reshape(-1), Q_vmin_percentile))
        Q_vmax = float(np.percentile(Q_cpu.reshape(-1), Q_vmax_percentile))
        if (not np.isfinite(Q_vmin)) or (not np.isfinite(Q_vmax)) or (Q_vmin >= Q_vmax) or (abs(Q_vmax - Q_vmin) < 1e-12):
            Q_vmin, Q_vmax = None, None

    # --------------------------
    # 4b) SANITY CHECKS (prints)
    # --------------------------
    def _stats(name, mat):
        flat = mat.reshape(-1)
        p = np.percentile(flat, [0, 1, 5, 50, 95, 99, 100])
        return (f"{name}: shape={mat.shape}, "
                f"min={p[0]:.6g}, p1={p[1]:.6g}, p5={p[2]:.6g}, "
                f"p50={p[3]:.6g}, p95={p[4]:.6g}, p99={p[5]:.6g}, max={p[6]:.6g}, "
                f"range={p[6]-p[0]:.6g}")

    sanity_lines = []
    sanity_lines.append(_stats("C_used(sorted)", C_used_cpu))
    sanity_lines.append(_stats("C_geom(sorted)", C_geom_cpu))
    sanity_lines.append(_stats("Q(sorted)", Q_cpu))

    # Detect “flat image” risk
    if np.std(C_used_cpu) < 1e-4:
        sanity_lines.append("WARNING: std(C_used) is very small -> heatmap may look almost constant (flat color).")

    # Does Q put mass on low-cost regions? (Using C_geom is often more meaningful)
    # Compare expected cost under Q vs uniform.
    eps = 1e-12
    Q_row = Q_cpu / (Q_cpu.sum(axis=1, keepdims=True) + eps)
    uniform = np.full_like(Q_row, 1.0 / Q_row.shape[1])
    E_cost_Q = (Q_row * C_geom_cpu).sum(axis=1).mean()
    E_cost_U = (uniform * C_geom_cpu).sum(axis=1).mean()
    sanity_lines.append(f"E[C_geom | Q] (mean over rows) = {E_cost_Q:.6g}")
    sanity_lines.append(f"E[C_geom | uniform] (mean over rows) = {E_cost_U:.6g}")
    sanity_lines.append(f"Gap (uniform - Q) = {E_cost_U - E_cost_Q:.6g}  (should be >= 0 if Q prefers low cost)")

    if sanity_print:
        print("\n".join(sanity_lines))

    # --------------------------
    # 5) Plot
    # --------------------------
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    ax0, ax1 = axes

    im0 = ax0.imshow(C_used_cpu, aspect="auto", interpolation="nearest", vmin=C_vmin, vmax=C_vmax)
    ax0.set_title("Cost matrix C_used (sorted)")
    ax0.set_xlabel("Prototypes (sorted)")
    ax0.set_ylabel("Batch samples (sorted)")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    im1 = ax1.imshow(Q_cpu, aspect="auto", interpolation="nearest", vmin=Q_vmin, vmax=Q_vmax)
    ax1.set_title("Assignment matrix Q (sorted)")
    ax1.set_xlabel("Prototypes (same order as C)")
    ax1.set_ylabel("Batch samples (same order as C)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    if title is None:
        title = f"C & Q (B={B}, K={K}) | rows={sort_rows}, cols={sort_cols}, mode={cost_mode}"
    fig.suptitle(title, y=0.98)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)

    # --------------------------
    # 6) Save CSVs (sorted matrices + indices)
    # --------------------------
    if save_csv_dir is  None:
        save_csv_dir = os.path.dirname(save_path) + f"/task_{task}_epoch_{epoch}"
         
    if True:
        os.makedirs(save_csv_dir, exist_ok=True)
        np.savetxt(os.path.join(save_csv_dir, f"{csv_prefix}_C_used_sorted.csv"), C_used_cpu, delimiter=",")
        np.savetxt(os.path.join(save_csv_dir, f"{csv_prefix}_C_geom_sorted.csv"), C_geom_cpu, delimiter=",")
        np.savetxt(os.path.join(save_csv_dir, f"{csv_prefix}_Q_sorted.csv"), Q_cpu, delimiter=",")
        np.savetxt(os.path.join(save_csv_dir, f"{csv_prefix}_row_idx.csv"), row_idx.astype(np.int64), fmt="%d", delimiter=",")
        np.savetxt(os.path.join(save_csv_dir, f"{csv_prefix}_col_idx.csv"), col_idx.astype(np.int64), fmt="%d", delimiter=",")
        with open(os.path.join(save_csv_dir, f"{csv_prefix}_sanity_report.txt"), "w") as f:
            f.write("\n".join(sanity_lines) + "\n")

    # return tensors for further analysis
    row_idx_t = torch.from_numpy(row_idx).to(device)
    col_idx_t = torch.from_numpy(col_idx).to(device)
    if True:
        def _sha1_f32(x: np.ndarray) -> str:
            x = np.asarray(x, dtype=np.float32, order="C")
            return hashlib.sha1(x.tobytes()).hexdigest()

        def _ensure_dir(p: str):
            os.makedirs(p, exist_ok=True)

        # 在 _plot_cost_matrix 内部，to numpy 之后、排序之前，加入：
        C_used_raw = C_used.detach().float().cpu().numpy()   # [B,K] 未排序
        C_geom_raw = C_geom.detach().float().cpu().numpy()   # [B,K] 未排序
        Q_raw      = Q.detach().float().cpu().numpy()        # [B,K] 未排序
        logits_raw = None
        if logits is not None:
            logits_raw = logits.detach().float().cpu().numpy()

        # （你原有排序逻辑不变，排序后得到 C_used_cpu / C_geom_cpu / Q_cpu）

        # 在保存 CSV 的部分，同时保存一个 npz bundle（强烈建议）
        if save_csv_dir is None:
            save_csv_dir = os.path.dirname(save_path) + f"_{task}_{epoch}"
        _ensure_dir(save_csv_dir)

        # 把 vmin/vmax 最终值也保存下来（否则重放时可能颜色不一致）
        bundle_meta = dict(
            B=int(B), K=int(K), D=int(D),
            cost_mode=cost_mode,
            normalize=bool(normalize),
            sort_rows=sort_rows,
            sort_cols=sort_cols,
            # 注意：保存 percentiles + 最终的 vmin/vmax
            C_vmin_percentile=C_vmin_percentile,
            C_vmax_percentile=C_vmax_percentile,
            Q_vmin_percentile=Q_vmin_percentile,
            Q_vmax_percentile=Q_vmax_percentile,
            C_vmin=float(C_vmin) if C_vmin is not None else None,
            C_vmax=float(C_vmax) if C_vmax is not None else None,
            Q_vmin=float(Q_vmin) if Q_vmin is not None else None,
            Q_vmax=float(Q_vmax) if Q_vmax is not None else None,
            dpi=int(dpi),
        )

        bundle_meta["sha1_C_used_raw"] = _sha1_f32(C_used_raw)
        bundle_meta["sha1_Q_raw"]      = _sha1_f32(Q_raw)

        # （可选）保存 logits 的 hash，方便确认“同一次 forward”
        if logits_raw is not None:
            bundle_meta["sha1_logits_raw"] = _sha1_f32(logits_raw)

        bundle_path = os.path.join(save_csv_dir, f"{csv_prefix}_bundle.npz")
        np.savez_compressed(
            bundle_path,
            # raw (unsorted)
            C_used_raw=C_used_raw,
            C_geom_raw=C_geom_raw,
            Q_raw=Q_raw,
            logits_raw=logits_raw if logits_raw is not None else np.array([], dtype=np.float32),

            # sorted
            C_used_sorted=C_used_cpu,
            C_geom_sorted=C_geom_cpu,
            Q_sorted=Q_cpu,

            # permutations
            row_idx=row_idx.astype(np.int64),
            col_idx=col_idx.astype(np.int64),

            # (Level B) 想做“重算 cost”验证才需要：保存 feats & protos
            # feats=batch_feats.detach().float().cpu().numpy(),
            # protos=proto_weights.detach().float().cpu().numpy(),
        )
        import json
        with open(os.path.join(save_csv_dir, f"{csv_prefix}_bundle_meta.json"), "w") as f:
            json.dump(bundle_meta, f, indent=2)

        print(f"[Debug] Saved bundle: {bundle_path}")
        print(f"[Debug] meta sha1 C_used_raw={bundle_meta['sha1_C_used_raw'][:10]} Q_raw={bundle_meta['sha1_Q_raw'][:10]}")

    return {
        "C_used": C_used.detach(),
        "C_geom": C_geom.detach(),
        "Q": Q.detach(),
        "row_idx": row_idx_t,
        "col_idx": col_idx_t,
        "C_used_sorted": torch.from_numpy(C_used_cpu).to(device),
        "C_geom_sorted": torch.from_numpy(C_geom_cpu).to(device),
        "Q_sorted": torch.from_numpy(Q_cpu).to(device),
    }

@torch.no_grad()
def _plot_cost_matrix2(
    batch_feats: torch.Tensor,               # [B, D]
    proto_weights: torch.Tensor,             # [K, D]
    *,
    Q: torch.Tensor,                         # [B, K]  (required here)
    logits: Optional[torch.Tensor] = None,   # [B, K] optional
    cost_mode: str = "neg_dot",              # "neg_dot" | "sqeuclidean"
    normalize: bool = True,                  # normalize feats/protos before dot product
    sort_rows: str = "argmin",               # "none" | "argmin" | "mincost"
    sort_cols: str = "mass",                 # "mass" | "usage" | "avgcost" | "none"
    title: Optional[str] = None,
    save_path: str = "./cost_and_q.png",
    dpi: int = 160,
    # C visualization clipping
    C_vmax_percentile: Optional[float] = 99.0,
    C_vmin_percentile: Optional[float] = 1.0,
    # Q visualization clipping
    Q_vmax_percentile: Optional[float] = 99.5,
    Q_vmin_percentile: Optional[float] = 0.0,  # usually 0 makes sense for Q
    show: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Plot side-by-side heatmaps:
      Left: cost matrix C (geometry)
      Right: assignment matrix Q (Sinkhorn plan)

    IMPORTANT:
      - Both plots share the SAME row/col ordering.
      - Row ordering: derived from C (default argmin(C)) so you can see whether Q mass
        lands on low-cost regions.
      - Column ordering: derived from Q (default by column mass sum_i Q_ik) so both x-axes match.

    Returns:
      - C (unsorted), Q (unsorted)
      - row_idx, col_idx (the shared ordering)
      - C_sorted, Q_sorted (for further analysis)
    """
    assert batch_feats.dim() == 2, f"batch_feats must be [B,D], got {batch_feats.shape}"
    assert proto_weights.dim() == 2, f"proto_weights must be [K,D], got {proto_weights.shape}"
    assert Q is not None, "Q must be provided to plot Q heatmap."
    assert Q.dim() == 2, f"Q must be [B,K], got {Q.shape}"

    device = batch_feats.device
    B, D = batch_feats.shape
    K, D2 = proto_weights.shape
    assert D == D2, f"Feature dim mismatch: batch {D}, protos {D2}"
    assert Q.shape[0] == B and Q.shape[1] == K, f"Q shape {Q.shape} must match [B,K]=[{B},{K}]"

    # --------------------------
    # 1) Build C (geometry cost)
    # --------------------------
    if logits is not None:
        # higher similarity => lower cost
        C = -logits
    else:
        z = batch_feats
        c = proto_weights

        if normalize and cost_mode == "neg_dot":
            z = torch.nn.functional.normalize(z, dim=1)
            c = torch.nn.functional.normalize(c, dim=1)

        if cost_mode == "neg_dot":
            C = -(z @ c.t())  # [B,K]
        elif cost_mode == "sqeuclidean":
            z2 = (batch_feats ** 2).sum(dim=1, keepdim=True)            # [B,1]
            c2 = (proto_weights ** 2).sum(dim=1, keepdim=True).t()      # [1,K]
            sim = batch_feats @ proto_weights.t()
            C = z2 + c2 - 2.0 * sim
        else:
            raise ValueError(f"Unknown cost_mode: {cost_mode}")

    # to numpy for sorting & plotting
    C_cpu = C.detach().float().cpu().numpy()      # [B,K]
    Q_cpu = Q.detach().float().cpu().numpy()      # [B,K]

    # --------------------------
    # 2) Row sorting (from C)
    # --------------------------
    row_idx = np.arange(B)
    if sort_rows == "argmin":
        row_best = np.argmin(C_cpu, axis=1)              # [B]
        row_minc = C_cpu[np.arange(B), row_best]         # [B]
        row_idx = np.lexsort((row_minc, row_best))       # group by best prototype, then by cost
    elif sort_rows == "mincost":
        row_minc = np.min(C_cpu, axis=1)
        row_idx = np.argsort(row_minc)
    elif sort_rows == "none":
        pass
    else:
        raise ValueError(f"Unknown sort_rows: {sort_rows}")

    C_cpu = C_cpu[row_idx, :]
    Q_cpu = Q_cpu[row_idx, :]

    # --------------------------
    # 3) Column sorting (prefer Q-derived)
    # --------------------------
    col_idx = np.arange(K)
    if sort_cols == "mass":
        # column mass from Q
        qmass = Q_cpu.sum(axis=0)              # [K]
        col_idx = np.argsort(-qmass)           # descending
    elif sort_cols == "usage":
        # hard usage based on argmax of Q (more faithful than argmin(C) for usage)
        hard = np.argmax(Q_cpu, axis=1)
        usage = np.bincount(hard, minlength=K).astype(np.float32)
        col_idx = np.argsort(-usage)
    elif sort_cols == "avgcost":
        col_avg = C_cpu.mean(axis=0)
        col_idx = np.argsort(col_avg)
    elif sort_cols == "none":
        pass
    else:
        raise ValueError(f"Unknown sort_cols: {sort_cols}")

    C_cpu = C_cpu[:, col_idx]
    Q_cpu = Q_cpu[:, col_idx]

    # --------------------------
    # 4) Robust scaling helpers
    # --------------------------
    def _robust_vmin_vmax(mat: np.ndarray,
                         vmin_p: Optional[float],
                         vmax_p: Optional[float]):
        if vmin_p is None or vmax_p is None:
            return None, None
        flat = mat.reshape(-1)
        vmin = float(np.percentile(flat, vmin_p))
        vmax = float(np.percentile(flat, vmax_p))
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin >= vmax):
            return None, None
        return vmin, vmax

    C_vmin, C_vmax = _robust_vmin_vmax(C_cpu, C_vmin_percentile, C_vmax_percentile)

    # For Q, often best to keep vmin=0; vmax clipped to avoid one-hot dominance
    if Q_vmin_percentile is None or Q_vmax_percentile is None:
        Q_vmin, Q_vmax = None, None
    else:
        Q_vmin = float(np.percentile(Q_cpu.reshape(-1), Q_vmin_percentile))
        Q_vmax = float(np.percentile(Q_cpu.reshape(-1), Q_vmax_percentile))
        if (not np.isfinite(Q_vmin)) or (not np.isfinite(Q_vmax)) or (Q_vmin >= Q_vmax):
            Q_vmin, Q_vmax = None, None

    # --------------------------
    # 5) Plot side-by-side
    # --------------------------
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    ax0, ax1 = axes

    im0 = ax0.imshow(C_cpu, aspect="auto", interpolation="nearest", vmin=C_vmin, vmax=C_vmax)
    ax0.set_title("Cost matrix C (sorted)")
    ax0.set_xlabel("Prototypes (sorted by Q)")
    ax0.set_ylabel("Batch samples (sorted)")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    im1 = ax1.imshow(Q_cpu, aspect="auto", interpolation="nearest", vmin=Q_vmin, vmax=Q_vmax)
    ax1.set_title("Assignment matrix Q (sorted)")
    ax1.set_xlabel("Prototypes (same order as C)")
    ax1.set_ylabel("Batch samples (same order as C)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    if title is None:
        title = f"C & Q (B={B}, K={K}) | rows={sort_rows}, cols={sort_cols}, mode={cost_mode}"
    fig.suptitle(title, y=0.98)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)

    # return tensors for further analysis
    row_idx_t = torch.from_numpy(row_idx).to(device)
    col_idx_t = torch.from_numpy(col_idx).to(device)

    return {
        "C": C.detach(),                                 # [B,K] original (unsorted)
        "Q": Q.detach(),                                 # [B,K] original (unsorted)
        "row_idx": row_idx_t,                             # [B]
        "col_idx": col_idx_t,                             # [K]
        "C_sorted": torch.from_numpy(C_cpu).to(device),   # [B,K] sorted view
        "Q_sorted": torch.from_numpy(Q_cpu).to(device),   # [B,K] sorted view
    }

def plot_cost_matrix(probe_images,model,args,path,task,epoch,data_index):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        feats, logits = model(probe_images)
        feats = torch.nn.functional.normalize(feats, dim=1)
    with torch.no_grad():
        Q,_ = distributed_sinkhorn(logits,epoch=epoch,total_epoch=args.epochs)   #    
    result = _plot_cost_matrix(
            batch_feats=feats,
            proto_weights=model.prototypes.weight,
            Q=Q,
            logits=logits,
            sort_rows="argmin",
            sort_cols="mass",
            save_path=f"{path}.png",
            task = task,
            epoch = epoch
        )
       
    save_probe_and_prototypes(
        probe_images,
        model,
        data_i=data_index,
        task_j=task,
        epoch_n=epoch,
        out_dir=os.path.dirname(path),
        Q=Q,
        save_logits=True,
        save_Q=True,
    )     
    if was_training:
        model.train()
    return result


import json
def save_probe_and_prototypes(
    probe_images: torch.Tensor,
    model: torch.nn.Module,
    *,
    data_i: int,
    task_j: int,
    epoch_n: int,
    out_dir: str,
    sample_ids: Optional[np.ndarray] = None,
    normalize_feats: bool = True,
    normalize_protos: bool = True,
    # NEW:
    Q: Optional[torch.Tensor] = None,                 # [B,K] if you already computed it
    sinkhorn_fn=None,                                # callable(logits)->Q, optional
    save_logits: bool = True,
    save_Q: bool = True,
    filename_ext: str = "npz",
) -> str:
    """
    Save feats/protos/logits/Q for probe_images (data_i) and current model (task_j, epoch_n)
    into data_{i}_task_{j}_epoch_{n}.npz
    """
    os.makedirs(out_dir, exist_ok=True)
    model_was_training = model.training
    model.eval()

    with torch.no_grad():
        out = model(probe_images)
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            feats = out[0]
            logits = out[1] if (len(out) >= 2) else None
        else:
            feats = out
            logits = None

        if not hasattr(model, "prototypes"):
            raise AttributeError("Model has no attribute 'prototypes' or 'prototypes.weight'.")
        protos_obj = getattr(model, "prototypes")
        protos = protos_obj.weight if hasattr(protos_obj, "weight") else protos_obj

        feats = feats.detach()
        protos = protos.detach()

        if normalize_feats:
            feats = torch.nn.functional.normalize(feats, dim=1)
        if normalize_protos:
            protos = torch.nn.functional.normalize(protos, dim=1)

        # Q (assignment)
        if save_Q:
            if Q is None:
                if logits is None or sinkhorn_fn is None:
                    raise ValueError("save_Q=True but Q is None and (logits or sinkhorn_fn) is missing.")
                Q = sinkhorn_fn(logits)
            Q_np = Q.detach().float().cpu().numpy()
        else:
            Q_np = None

        payload = {
            "feats": feats.float().cpu().numpy(),     # [B,D]
            "protos": protos.float().cpu().numpy(),   # [K,D]
            "data_i": int(data_i),
            "task_j": int(task_j),
            "epoch_n": int(epoch_n),
            "normalize_feats": bool(normalize_feats),
            "normalize_protos": bool(normalize_protos),
         }

        if sample_ids is not None:
            sample_ids = np.asarray(sample_ids)
            assert sample_ids.shape[0] == payload["feats"].shape[0]
            payload["sample_ids"] = sample_ids

        if save_logits and (logits is not None):
            payload["logits"] = logits.detach().float().cpu().numpy()  # [B,K]

        if save_Q and (Q_np is not None):
            payload["Q"] = Q_np  # [B,K]

    if model_was_training:
        model.train()

    fname = f"data_{data_i}_task_{task_j}_epoch_{epoch_n}.{filename_ext}"
    fpath = os.path.join(out_dir, fname)
    np.savez_compressed(fpath, **payload)
    return fpath


if __name__ == "__main__":
    main()
