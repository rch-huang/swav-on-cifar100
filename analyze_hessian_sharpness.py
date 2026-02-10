import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def collect_records(root_dir, block="theta", basis="anchor"):
    records = []

    task_dirs = sorted(glob.glob(os.path.join(root_dir, "task_*")))
    for task_dir in task_dirs:
        task_id = int(os.path.basename(task_dir).split("_")[1])
        proj_path = os.path.join(task_dir, "step_projections", "proj.jsonl")
        if not os.path.isfile(proj_path):
            continue

        with open(proj_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("block") != block:
                    continue
                if rec.get("basis") != basis:
                    continue
                rec["_task_id"] = task_id
                records.append(rec)

    # anchor order
    records.sort(key=lambda r: (r["_task_id"], r["epoch"], r["step"]))
    return records


def plot_stacked_heatmap(
    root_dir,
    block="theta",
    basis="anchor",
    topk=None,
    figsize=(20, 6),
    cmap_name="viridis",
    save_path=None,
):
    records = collect_records(root_dir, block=block, basis=basis)
    assert len(records) > 0, "No valid records found."

    # determine K
    K = len(records[0]["proj"])
    if topk is None:
        topk = K
    topk = min(topk, K)

    # build heights
    heights_all = []
    task_ids = []
    epochs = []

    for rec in records:
        proj = np.asarray(rec["proj"][:topk])
        eig = np.asarray(rec["eigvals"][:topk])
        denom = rec["denom_rayleigh"]

        h = eig * proj**2 / denom
        heights_all.append(h)
        task_ids.append(rec["_task_id"])
        epochs.append(rec["epoch"])

    heights_all = np.asarray(heights_all)
    N = heights_all.shape[0]
    x = np.arange(N)

    # colors mapped by eig index
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=1, vmax=topk)
    colors = [cmap(norm(i + 1)) for i in range(topk)]

    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(N)

    for i in range(topk):
        ax.bar(
            x,
            heights_all[:, i] * 100.0,
            bottom=bottom * 100.0,
            width=1.0,
            color=colors[i],
            edgecolor="none",
        )
        bottom += heights_all[:, i]

    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Hessian energy percentage (%)")
    ax.set_xlabel("Anchor index")

    ax.set_title(f"Stacked Hessian-energy heatmap ({block}, top-{topk})")

    # -------------------------------------------------
    # (1) Task boundary vertical lines
    # -------------------------------------------------
    last_task = task_ids[0]
    for i, t in enumerate(task_ids):
        if t != last_task:
            ax.axvline(i - 0.5, color="black", linestyle="--", linewidth=1)
            last_task = t

    # -------------------------------------------------
    # (2) Anchor → epoch mapping (secondary x-axis)
    # -------------------------------------------------
    unique_epochs = sorted(set(epochs))
    epoch_to_x = {}
    for ep in unique_epochs:
        idxs = [i for i, e in enumerate(epochs) if e == ep]
        epoch_to_x[ep] = int(np.mean(idxs))

    ax2 = ax.secondary_xaxis("top")
    ax2.set_xticks(list(epoch_to_x.values()))
    ax2.set_xticklabels([f"ep{ep}" for ep in epoch_to_x.keys()])
    ax2.set_xlabel("Epoch")

    # -------------------------------------------------
    # (3) Eig-index colorbar (right side)
    # -------------------------------------------------
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        pad=0.01,
        fraction=0.025,
    )
    cbar.set_label("Eig index")
    cbar.set_ticks([1, topk])
    cbar.set_ticklabels(["1st", f"{topk}th"])

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        print(f"[Saved] {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=str)
    parser.add_argument("--block", type=str, default="theta")
    parser.add_argument("--basis", type=str, default="anchor")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--save", type=str, default=None)

    args = parser.parse_args()

    plot_stacked_heatmap(
        root_dir=args.root_dir,
        block=args.block,
        basis=args.basis,
        topk=args.topk,
        save_path=os.path.join(args.root_dir, f"hessian_energy_{args.block}_{args.basis}_top{args.topk}.png"),
    )

 
