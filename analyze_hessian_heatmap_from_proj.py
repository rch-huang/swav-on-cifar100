#!/usr/bin/env python3
"""
Plot sharpness curves + stacked heatmap from proj.jsonl across ALL task_* under a root.

Added:
  energy_mode = "dir"
    e_i = (proj_i)^2
  This aligns EXACTLY with tracker/controller theta/anchor_ratio.

Original modes preserved:
  pos / abs / signed
"""

import os
import json
import glob
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# utils
# ============================================================

def _natural_task_id(task_dir: str) -> int:
    base = os.path.basename(task_dir.rstrip("/"))
    try:
        return int(base.split("_")[1])
    except Exception:
        return 10**9


def load_all_records(root_dir: str) -> List[Dict[str, Any]]:
    task_dirs = sorted(glob.glob(os.path.join(root_dir, "task_*")), key=_natural_task_id)
    all_recs: List[Dict[str, Any]] = []

    for td in task_dirs:
        tid = _natural_task_id(td)
        p = os.path.join(td, "step_projections", "proj.jsonl")
        if not os.path.isfile(p):
            continue
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r["_task_id"] = tid
                all_recs.append(r)

    def _key(r):
        return (
            int(r.get("_task_id", 10**9)),
            int(r.get("epoch", -1)),
            int(r.get("step", -1)),
            int(r.get("anchor_epoch", -1)),
        )

    all_recs.sort(key=_key)
    return all_recs


def filter_records(
    records: List[Dict[str, Any]],
    *,
    block: str,
    basis: str,
) -> List[Dict[str, Any]]:
    out = []
    for r in records:
        if r.get("block") != block:
            continue
        if r.get("basis") != basis:
            continue
        if "proj" not in r:
            continue
        out.append(r)
    return out


def build_anchor_index_and_task_boundaries(
    records: List[Dict[str, Any]]
) -> Tuple[np.ndarray, List[int], List[int]]:
    N = len(records)
    x = np.arange(N)
    task_ids = [int(r.get("_task_id", -1)) for r in records]

    boundaries = []
    if N > 0:
        last = task_ids[0]
        for i, t in enumerate(task_ids):
            if t != last:
                boundaries.append(i)
                last = t
    return x, boundaries, task_ids


# ============================================================
# sharpness
# ============================================================

def compute_sharpness_arrays(
    records: List[Dict[str, Any]],
    topk_avg: int,
) -> Tuple[np.ndarray, np.ndarray]:
    top1 = []
    avgk = []
    for r in records:
        lam = np.asarray(r["eigvals"], dtype=np.float64)
        if lam.size == 0:
            top1.append(np.nan)
            avgk.append(np.nan)
            continue
        top1.append(float(lam[0]))
        k = min(int(topk_avg), lam.size)
        avgk.append(float(np.mean(lam[:k])))
    return np.asarray(top1), np.asarray(avgk)


# ============================================================
# energy definitions  ⭐ 核心修改点
# ============================================================

def energy_from_lam_proj(
    lam: np.ndarray,
    proj: np.ndarray,
    mode: str,
) -> np.ndarray:
    """
    energy definitions:

    dir    : proj^2                      (controller / tracker aligned)
    pos    : max(lam,0) * proj^2
    abs    : |lam| * proj^2
    signed : lam * proj^2
    """
    if mode == "dir":
        return proj ** 2

    if mode == "pos":
        lam_eff = np.maximum(lam, 0.0)
    elif mode == "abs":
        lam_eff = np.abs(lam)
    elif mode == "signed":
        lam_eff = lam
    else:
        raise ValueError(f"Unknown energy_mode={mode}")

    return lam_eff * (proj ** 2)


def build_stacked_matrix(
    records: List[Dict[str, Any]],
    m_stack: int,
    energy_mode: str,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Build heights [N, m_stack].

    For energy_mode="dir":
      denom = sum proj^2   (ALL available proj dims)

    For others:
      denom = sum lambda * proj^2
    """
    N = len(records)
    H = np.zeros((N, m_stack), dtype=np.float64)

    for i, r in enumerate(records):
        proj = np.asarray(r.get("proj", []), dtype=np.float64)
        lam = np.asarray(r.get("eigvals", []), dtype=np.float64)

        if energy_mode == "dir":
            K = len(proj)
            if K == 0:
                continue
            e = proj[:K] ** 2
        else:
            K = min(len(proj), len(lam))
            if K == 0:
                continue
            e = energy_from_lam_proj(lam[:K], proj[:K], energy_mode)

        denom = float(np.sum(e))
        if not np.isfinite(denom) or denom < eps:
            continue

        m = min(int(m_stack), K)
        H[i, :m] = e[:m] / denom

    return H


# ============================================================
# plotting
# ============================================================

def plot_sharpness(
    records: List[Dict[str, Any]],
    *,
    topk_avg: int,
    block: str,
    basis: str,
    save_path: str,
):
    x, boundaries, _ = build_anchor_index_and_task_boundaries(records)
    top1, avgk = compute_sharpness_arrays(records, topk_avg=topk_avg)

    plt.figure(figsize=(20, 5))
    plt.plot(x, top1, label="top1: λ₁")
    plt.plot(x, avgk, label=f"avg_top{topk_avg}")
    for b in boundaries:
        plt.axvline(b - 0.5, linestyle="--", linewidth=1)

    plt.xlabel("Anchor index")
    plt.ylabel("Sharpness (eigenvalue)")
    plt.title(f"Sharpness curves (backbone weights)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Saved] {save_path}")


def plot_stacked_heatmap(
    records,
    *,
    m_stack: int,
    energy_mode: str,
    block: str,
    basis: str,
    save_path: str,
    gap: int = 15,
    max_marks: int = 6,
):
    records = list(records)
    N = len(records)

    x, boundaries, task_ids = build_anchor_index_and_task_boundaries(records)

    K_global = max(len(r.get("proj", [])) for r in records)

    H = build_stacked_matrix(
        records,
        m_stack=m_stack,
        energy_mode=energy_mode,
    )

    # ---------- global contribution (for colorbar marks) ----------
    E_global = np.zeros(K_global, dtype=np.float64)
    for r in records:
        proj = np.asarray(r.get("proj", []), dtype=np.float64)
        lam = np.asarray(r.get("eigvals", []), dtype=np.float64)

        if energy_mode == "dir":
            K = len(proj)
            if K == 0:
                continue
            e = proj[:K] ** 2
        else:
            K = min(len(proj), len(lam))
            if K == 0:
                continue
            e = energy_from_lam_proj(lam[:K], proj[:K], energy_mode)

        E_global[:K] += e

    order = np.argsort(E_global)[::-1]

    marks = {1, K_global}
    for idx0 in order:
        idx = idx0 + 1
        if all(abs(idx - m0) >= gap for m0 in marks):
            marks.add(idx)
        if len(marks) >= max_marks:
            break
    marks = sorted(marks)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=1, vmax=K_global)

    fig, ax = plt.subplots(figsize=(22, 6))

    bottom = np.zeros(N, dtype=np.float64)
    for i in range(H.shape[1]):
        color = cmap(norm(i + 1))
        ax.bar(
            x,
            H[:, i] * 100.0,
            bottom=bottom * 100.0,
            width=1.0,
            color=color,
            edgecolor="none",
        )
        bottom += H[:, i]

    for b in boundaries:
        ax.axvline(b - 0.5, linestyle="--", linewidth=1, color="black", alpha=0.6)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Anchor index")

    ax.set_title(
        f"Stacked projection heatmap (backbone weights)\n"
        f"top-{m_stack}"
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Eigenvector index", rotation=270, labelpad=18)
    cbar.set_ticks(marks)
    cbar.set_ticklabels([str(i) for i in marks])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[Saved] {save_path}")


# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root_dir", type=str)
    ap.add_argument("--block", type=str, default="theta", choices=["theta", "C"])
    ap.add_argument("--basis", type=str, default="anchor")
    ap.add_argument("--topk_avg", type=int, default=10)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument(
        "--energy_mode",
        type=str,
        default="pos",
        choices=["dir", "pos", "abs", "signed"],
        help='dir = align with tracker/controller (proj^2)',
    )
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or args.root_dir
    os.makedirs(out_dir, exist_ok=True)

    all_recs = load_all_records(args.root_dir)
    recs = filter_records(all_recs, block=args.block, basis=args.basis)

    if not recs:
        raise RuntimeError("No matching records found.")

    sharp_path = os.path.join(
        out_dir,
        f"sharpness_{args.block}_{args.basis}_topk{args.topk_avg}.png"
    )
    heat_path = os.path.join(
        out_dir,
        f"stacked_heatmap_{args.block}_{args.basis}_m{args.m}_energy{args.energy_mode}.png"
    )

    plot_sharpness(
        recs,
        topk_avg=args.topk_avg,
        block=args.block,
        basis=args.basis,
        save_path=sharp_path,
    )

    plot_stacked_heatmap(
        recs,
        m_stack=args.m,
        energy_mode=args.energy_mode,
        block=args.block,
        basis=args.basis,
        save_path=heat_path,
    )


if __name__ == "__main__":
    main()
