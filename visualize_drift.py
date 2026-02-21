import os
import json
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def collect_drift_logs(log_root):
    """
    Traverse log_root and collect all drift logs.
    """
    pattern = os.path.join(
        log_root,
        "task_*",
        "task_optimal",
        "epoch_*",
        "drift_logs",
        "drift_*.json"
    )

    files = glob.glob(pattern)
    records = []

    for fpath in files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        task = data.get("task", None)
        epoch = data.get("epoch", None)
        step = data.get("step", None)

        blocks = data.get("blocks", {})

        for block_name, block_data in blocks.items():
            rec = {
                "task": task,
                "epoch": epoch,
                "step": step,
                "block": block_name,
                "overlap": block_data.get("overlap"),
                "min_singular": block_data.get("min_singular"),
                "mean_singular": block_data.get("mean_singular"),
                "eigval_ratio": block_data.get("eigval_ratio"),
                "frobenius_diff": block_data.get("frobenius_diff"),
            }
            records.append(rec)

    return records


def plot_metric(records, metric_name, save_dir):
    """
    Plot a specific metric vs epoch for theta and C separately.
    """

    by_block = defaultdict(list)

    for r in records:
        if r[metric_name] is None:
            continue
        by_block[r["block"]].append(r)

    for block, recs in by_block.items():
        recs = sorted(recs, key=lambda x: (x["epoch"], x["step"]))

        epochs = [r["epoch"] for r in recs]
        steps = [r["step"] for r in recs]
        values = [r[metric_name] for r in recs]

        # combine epoch + step for x-axis ordering
        x = [e + s * 1e-3 for e, s in zip(epochs, steps)]

        plt.figure(figsize=(8, 5))
        plt.plot(x, values, marker="o")
        plt.xlabel("Epoch (step shown as small offset)")
        plt.ylabel(metric_name)
        plt.title(f"{block} - {metric_name}")
        plt.grid(True)

        save_path = os.path.join(save_dir, f"{block}_{metric_name}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", type=str, required=True,
                        help="Path to log_* directory")
    args = parser.parse_args()

    log_root = args.log_root

    records = collect_drift_logs(log_root)

    if len(records) == 0:
        print("No drift logs found.")
        return

    save_dir = os.path.join(log_root, "drift_plots")
    os.makedirs(save_dir, exist_ok=True)

    metrics = [
        "overlap",
        "min_singular",
        "mean_singular",
        "eigval_ratio",
        "frobenius_diff",
    ]

    for m in metrics:
        plot_metric(records, m, save_dir)

    print("Done.")


if __name__ == "__main__":
    main()
