#!/usr/bin/env python3
"""
Plot ONE curve: total projection energy over top-K_big across all tasks & anchors
from a single proj.jsonl file.

- Reads records written by _process_step_delta():
    fields: task, epoch, step, anchor_epoch, block, basis, window_suffix, proj, eigvals, delta_norm2
- Computes per-record total energy:
    E_topK = sum(proj_i^2)                (energy_mode="dir"/"pos")
    E_topK = sum(abs(proj_i)^2)           (same as above)
    E_topK = sum((proj_i)^2) with sign??  (not meaningful for energy; provided for compatibility)

This script aggregates ONE value per (task, anchor_epoch) by mean over steps/epochs
and plots anchor index in chronological order.

Usage:
  python3 plot_total_energy_single_curve.py --jsonl /path/to/proj.jsonl --block theta --basis anchor --window window2

Outputs:
  - PNG next to jsonl (or --out_dir)
  - Optional: print per-anchor values (--print_values)
"""
import os, json, argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True, help="Path to proj.jsonl")
    ap.add_argument("--block", type=str, default="theta", choices=["theta","C"])
    ap.add_argument("--basis", type=str, default="anchor",
                    help='Exact match. Examples: "anchor", "prev_task_0"')
    ap.add_argument("--window", type=str, default="window2",
                    help='window_suffix to select, e.g. "window1" or "window2"')
    ap.add_argument("--energy_mode", type=str, default="dir",
                    choices=["dir","pos","abs","signed","lambda"],
                    help=(
                        "How to map proj -> scalar. "
                        "dir/pos/abs: sum(proj^2). "
                        "lambda: sum(eigvals * proj^2) (eigenvalue-weighted). "
                        "signed: sum(proj) (debug only; can be negative)."
                    ))
    ap.add_argument("--agg", type=str, default="mean", choices=["mean","median","p90","p99","max"],
                    help="Aggregation over records within the same (task, anchor_epoch).")
    ap.add_argument("--out_dir", type=str, default=None, help="Where to save png. Default: directory of jsonl.")
    ap.add_argument("--print_values", action="store_true", help="Print (anchor_index, task, anchor_epoch, value).")
    ap.add_argument("--show", action="store_true", help="Show the plot window.")
    return ap.parse_args()

def scalar_from_record(proj: np.ndarray, eigvals: np.ndarray | None, mode: str) -> float:
    """Map one record to a scalar summary.

    - dir/pos/abs: unweighted energy sum(proj^2)
    - lambda: eigenvalue-weighted energy sum(eigvals * proj^2)
    - signed: sum(proj) (debug only)
    """
    if mode in ("dir", "pos", "abs"):
        return float(np.sum(proj * proj))
    if mode == "lambda":
        if eigvals is None:
            raise ValueError("energy_mode=lambda requires 'eigvals' in each jsonl record")
        k = min(proj.shape[0], eigvals.shape[0])
        return float(np.sum(eigvals[:k] * (proj[:k] * proj[:k])))
    if mode == "signed":
        return float(np.sum(proj))
    raise ValueError(mode)

def agg_values(vals, how: str) -> float:
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return float("nan")
    if how == "mean":
        return float(np.mean(vals))
    if how == "median":
        return float(np.median(vals))
    if how == "p90":
        return float(np.quantile(vals, 0.90))
    if how == "p99":
        return float(np.quantile(vals, 0.99))
    if how == "max":
        return float(np.max(vals))
    raise ValueError(how)

def main():
    args = parse_args()
    jsonl_path = args.jsonl
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(jsonl_path)

    # (task, anchor_epoch) -> list of per-record energies
    buckets = defaultdict(list)

    # also keep a representative (epoch,step) count
    counts = defaultdict(int)

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("block") != args.block:
                continue
            if r.get("basis") != args.basis:
                continue
            if r.get("window_suffix") != args.window:
                continue

            proj = np.asarray(r["proj"], dtype=np.float64)
            eigvals = None
            if args.energy_mode == "lambda":
                eigvals = np.asarray(r.get("eigvals", []), dtype=np.float64)
            E = scalar_from_record(proj, eigvals, args.energy_mode)

            key = (int(r["task"]), int(r["anchor_epoch"]))
            buckets[key].append(E)
            counts[key] += 1

    if not buckets:
        raise RuntimeError(
            f"No records matched block={args.block}, basis={args.basis}, window={args.window} in {jsonl_path}"
        )

    # sort anchors in chronological order: by task then anchor_epoch
    keys = sorted(buckets.keys(), key=lambda x: (x[0], x[1]))
    y = [agg_values(buckets[k], args.agg) for k in keys]

    # x = global anchor index
    x = np.arange(len(keys))

    # plot
    plt.figure(figsize=(14, 4))
    plt.plot(x, y)
    plt.xlabel("Global anchor index (sorted by task, anchor_epoch)")
    plt.ylabel(f"Total top-K_big projection ({args.agg})")
    plt.title(f"Total projection energy curve | block={args.block}, basis={args.basis}, window={args.window}, mode={args.energy_mode}")

    # mark task boundaries
    last_task = keys[0][0]
    for i, (t, a) in enumerate(keys):
        if t != last_task:
            plt.axvline(i - 0.5, linestyle="--", linewidth=1)
            last_task = t

    plt.tight_layout()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(jsonl_path))
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(
        out_dir,
        f"total_energy_singlecurve_{args.block}_{args.basis}_{args.window}_{args.energy_mode}_{args.agg}.png"
    )
    plt.savefig(out_png, dpi=200)

    if args.print_values:
        for i, k in enumerate(keys):
            t, a = k
            print(f"{i}\ttask={t}\tanchor_epoch={a}\tN={counts[k]}\tvalue={y[i]:.6g}")

    if args.show:
        plt.show()
    else:
        plt.close()

    print(f"[OK] Saved: {out_png}")

if __name__ == "__main__":
    main()