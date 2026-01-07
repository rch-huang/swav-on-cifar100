#!/usr/bin/env python3
import argparse
import re
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Regex patterns
# =========================
PAT_TASK  = re.compile(r"Starting training on task\s+(\d+)")
PAT_EPOCH = re.compile(r"\bEpoch\s+(\d+)\b")
PAT_ROWCOL = re.compile(r"min_row_sum=([0-9eE\+\-\.]+),\s*min_col_sum=([0-9eE\+\-\.]+)")
PAT_GAP = re.compile(r"q_top1_top2_gap=([0-9eE\+\-\.]+)")

def to_float(x: str) -> float:
    m = re.search(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", x)
    return float(m.group(0))

# =========================
# Parse log
# =========================
def parse_log(path: str) -> pd.DataFrame:
    lines = pathlib.Path(path).read_text(errors="ignore").splitlines()

    cur_task = 0
    cur_epoch = None
    rows = []
    rec = None

    for ln in lines:
        mt = PAT_TASK.search(ln)
        if mt:
            cur_task = int(mt.group(1))
            cur_epoch = None
            rec = None

        me = PAT_EPOCH.search(ln)
        if me:
            cur_epoch = int(me.group(1))
            rec = {"task": cur_task, "epoch": cur_epoch}

        mrc = PAT_ROWCOL.search(ln)
        if mrc and cur_epoch is not None:
            if rec is None:
                rec = {"task": cur_task, "epoch": cur_epoch}
            rec["min_row_sum"] = to_float(mrc.group(1))
            rec["min_col_sum"] = to_float(mrc.group(2))
            rows.append(rec.copy())

        mg = PAT_GAP.search(ln)
        if mg and cur_epoch is not None:
            if rec is None:
                rec = {"task": cur_task, "epoch": cur_epoch}
            rec["gap"] = to_float(mg.group(1))
            rows.append(rec.copy())

    if not rows:
        raise RuntimeError(f"No usable data found in {path}")

    df = pd.DataFrame(rows)
    return df.groupby(["task", "epoch"]).mean().reset_index()

# =========================
# Global epoch
# =========================
def add_global_epoch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    offsets = {}
    cum = 0
    for t, mx in df.groupby("task")["epoch"].max().sort_index().items():
        offsets[t] = cum
        cum += int(mx) + 1
    df["global_epoch"] = df.apply(
        lambda r: offsets[int(r["task"])] + int(r["epoch"]), axis=1
    )
    return df.sort_values("global_epoch")

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+",
        help="Pairs of LABEL LOGFILE")
    ap.add_argument("--xmax", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", type=str, default="Btype_compare.png")
    args = ap.parse_args()

    if len(args.pairs) % 2 != 0:
        raise ValueError("Pairs must be LABEL LOGFILE")

    series = []
    for i in range(0, len(args.pairs), 2):
        label = args.pairs[i]
        path  = args.pairs[i + 1]
        df = add_global_epoch(parse_log(path))
        df = df[df["global_epoch"] <= args.xmax]
        series.append((label, df))

    # =========================
    # Plot
    # =========================
    fig, ax1 = plt.subplots(figsize=(12, 6))

    Y1_MIN, Y1_MAX = -8.0, -0.1
    Y2_MIN, Y2_MAX = 0.0, 0.9

    line_colors = []

    # Left axis: log10(min_row_sum)
    for label, df in series:
        line, = ax1.plot(
            df["global_epoch"],
            np.log10(df["min_row_sum"]),
            linewidth=2
        )
        line_colors.append(line.get_color())

    ax1.set_xlabel("Global Epoch")
    ax1.set_ylabel("log10(min_row_sum)")
    ax1.set_ylim(Y1_MIN, Y1_MAX)
    ax1.set_xticks(np.arange(0, args.xmax + 1, args.epochs))
    ax1.grid(True, alpha=0.3)

    # Right axis: gap
    ax2 = ax1.twinx()
    for (_, df), color in zip(series, line_colors):
        if "gap" in df.columns:
            ax2.plot(
                df["global_epoch"],
                df["gap"],
                linestyle="--",
                color=color,
                alpha=0.8,
                linewidth=1
            )

    ax2.set_ylabel("Top1–Top2 gap")
    ax2.set_ylim(Y2_MIN, Y2_MAX)

    # =========================
    # Bottom labels (figure-level)
    # =========================
    n = len(series)
    y_text = -0.01        # below plot
    x_start = 0.5 - 0.12 * (n - 1)
    dx = 0.24

    for i, ((label, _), color) in enumerate(zip(series, line_colors)):
        fig.text(
            x_start + i * dx,
            y_text,
            label,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=color
        )

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[Saved] {args.out}")

if __name__ == "__main__":
    main()
