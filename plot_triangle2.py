import os
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Utilities
# ============================================================

@dataclass(frozen=True)
class FileKey:
    data_i: int
    task_j: int
    epoch_n: int
    path: str


_FILENAME_RE = re.compile(
    r"data_(?P<i>\d+)_task_(?P<j>\d+)_epoch_(?P<n>\d+)\.(?P<ext>.+)$"
)


def scan_logged_files(log_dir: str) -> List[FileKey]:
    items: List[FileKey] = []
    for fn in os.listdir(log_dir):
        m = _FILENAME_RE.match(fn)
        if not m:
            continue
        items.append(
            FileKey(
                int(m.group("i")),
                int(m.group("j")),
                int(m.group("n")),
                os.path.join(log_dir, fn),
            )
        )
    items.sort(key=lambda x: (x.data_i, x.task_j, x.epoch_n))
    return items


def load_npz(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    if "meta_json" in d:
        try:
            d["meta_json"] = json.loads(str(d["meta_json"]))
        except Exception:
            pass
    return d


# ============================================================
# Ordering logic (MATCHES TRAINING-TIME SEMANTICS)
# ============================================================

def proto_order_from_qmass(Q: np.ndarray) -> np.ndarray:
    """
    Prototype order = descending Sinkhorn mass.
    Q: [B, K]
    """
    qmass = Q.sum(axis=0)
    return np.argsort(-qmass).astype(np.int64)


def row_order_from_C_argmin(C_used: np.ndarray) -> np.ndarray:
    """
    Row order identical to training-time sort_rows="argmin":
    primary: argmin prototype
    secondary: min cost
    """
    B = C_used.shape[0]
    best = np.argmin(C_used, axis=1)
    minc = C_used[np.arange(B), best]
    return np.lexsort((minc, best)).astype(np.int64)


def get_C_used(d: Dict[str, Any]) -> np.ndarray:
    """
    Training-time definition:
    C_used = -logits if logits exist, otherwise geometric fallback.
    """
    if "logits" in d:
        return -np.asarray(d["logits"], dtype=np.float32)

    # geometric fallback (should not happen if you saved logits)
    feats = d["feats"]
    protos = d["protos"]
    return -(feats @ protos.T).astype(np.float32)


# ============================================================
# Main plotting function
# ============================================================

def plot_cost_triangle(
    log_dir: str,
    out_png: str,
    *,
    dpi: int = 220*2,
    a4_landscape: bool = True,
    vmin_percentile: float = 1.0,
    vmax_percentile: float = 99.0,
):
    import os
    #check train_log file in log_dir
    train_log_path = os.path.join(log_dir, "train_log.txt")
    if not os.path.isfile(train_log_path):
        raise RuntimeError(f"train_log.txt not found in {log_dir}")
    else:
        # check the epislon and temp values in first row of train_log.txt, like "Namespace(..., temperature=0.1, epsilon=0.02, ...)"
        with open(train_log_path, "r") as f:
            first_line = f.readline()
            if "epsilon=" in first_line and "temperature=" in first_line:
                epsilon_match = re.search(r"epsilon=([0-9]*\.?[0-9]+)", first_line)
                temperature_match = re.search(r"temperature=([0-9]*\.?[0-9]+)", first_line)
                if epsilon_match and temperature_match:
                    epsilon_value = float(epsilon_match.group(1))
                    temperature_value = float(temperature_match.group(1))
                    print(f"Found in train_log.txt: epsilon={epsilon_value}, temperature={temperature_value}")
                else:
                    raise RuntimeError("Could not parse epsilon or temperature from train_log.txt")
            else:
                raise RuntimeError("epsilon or temperature not found in train_log.txt")
    """
    Outputs two PNGs:
      - *_C.png : Top row = Q-row (diagonal Q only), below = C triangle
      - *_Q.png : Only the Q-row (diagonal Q only)

    Ordering:
      - proto order for task j: from Q mass of data_j_task_j_epoch_last
      - sample order for data i: from C_used argmin of data_i_task_i_epoch_last
      - Q-row uses ONLY Q_{jj}^{(n)} with proto order π_j
      - C triangle uses proto order π_j and sample order of data i

    Display:
      - y-ticks only on each C-row's first cell (data_i_task_i_epoch_first), multiples of 50
      - titles only on top Q-row to save space
      - geometry-driven fig size avoids the “huge row spacing” bug
    """

    # -----------------------------
    # scan and index files
    # -----------------------------
    items = scan_logged_files(log_dir)
    if not items:
        raise RuntimeError(f"No data files found in {log_dir}")

    tasks = sorted({x.task_j for x in items})
    J = max(tasks) + 1

    epochs_per_task = {j: sorted({x.epoch_n for x in items if x.task_j == j}) for j in range(J)}
    first_epoch = {j: v[0] for j, v in epochs_per_task.items() if v}
    last_epoch  = {j: v[-1] for j, v in epochs_per_task.items() if v}

    file_map = {(x.data_i, x.task_j, x.epoch_n): x.path for x in items}

    # -----------------------------
    # infer matrix aspect H/K from data
    # -----------------------------
    ratios = []
    for (i, j, n), p in file_map.items():
        if i > j:
            continue
        d = load_npz(p)
        if "logits" in d:
            H, K = d["logits"].shape
        elif "Q" in d:
            H, K = d["Q"].shape
        else:
            continue
        ratios.append(H / K)
    if not ratios:
        raise RuntimeError("Cannot infer matrix aspect ratio.")
    matrix_aspect = float(np.median(ratios))

    # -----------------------------
    # proto order π_j from Q_{jj}^{last}
    # -----------------------------
    # ============================================================
    # proto order: GLOBAL (from data_0_task_0_epoch_last)
    # ============================================================

    # -----------------------------
    # proto order π_global from Q_{00}^{last}
    # -----------------------------
    if 0 not in last_epoch:
        raise RuntimeError("Cannot build global proto order: task 0 has no epochs.")

    key = (0, 0, last_epoch[0])
    if key not in file_map:
        raise RuntimeError(f"Missing file for global proto order: {key}")

    d0 = load_npz(file_map[key])
    if "Q" not in d0:
        raise KeyError(f"Missing Q in {file_map[key]} for global proto order.")

    if True:
        # instead of: proto_order_global = proto_order_from_qmass(d0["Q"])

        Q0 = d0["Q"]
        C0 = get_C_used(d0)
        proto_score = (Q0 * C0).sum(axis=0)   # column expectation
        proto_order_global = np.argsort(proto_score)


    else:
        proto_order_global = proto_order_from_qmass(d0["Q"])


    # -----------------------------
    # sample order per data i from C_{ii}^{last}
    # -----------------------------
    sample_order_by_data = {}
    for i in range(J):
        if i not in last_epoch:
            continue
        key = (i, i, last_epoch[i])
        if key not in file_map:
            continue
        d = load_npz(file_map[key])
        sample_order_by_data[i] = row_order_from_C_argmin(get_C_used(d))

    # -----------------------------
    # column layout: task blocks
    # -----------------------------
    task_col_start = {}
    col = 0
    for j in range(J):
        task_col_start[j] = col
        col += len(epochs_per_task.get(j, []))
    total_cols = col
    if total_cols == 0:
        raise RuntimeError("No epochs found.")
    # total rows for C-figure = 1 (Q-row) + J (C rows)
    total_rows_C = 1 + J
    total_rows_Q = 1

    # -----------------------------
    # geometry-driven figure size (avoid spacing bug)
    # NOTE: width determined by total_cols, height by subfig_h*rows
    # -----------------------------
    A4_W = 11.69 if a4_landscape else 8.27
    A4_H = 8.27  if a4_landscape else 11.69

    left_margin   = 0.55
    right_margin  = 0.18
    top_margin    = 0.60
    bottom_margin = 0.45

    row_spacing_in = 0.04  # very small physical spacing

    usable_w = A4_W - left_margin - right_margin
    subfig_w = usable_w / total_cols
    subfig_h = subfig_w * matrix_aspect

    def compute_figsize(nrows):
        content_h = nrows * subfig_h + (nrows - 1) * row_spacing_in + top_margin + bottom_margin
        fig_w = A4_W
        fig_h = min(content_h, A4_H)  # do NOT force fill; just cap if too tall
        return (fig_w, fig_h)

    figsize_C = compute_figsize(total_rows_C)
    figsize_Q = compute_figsize(total_rows_Q)

    # -----------------------------
    # y-ticks multiples of 50
    # -----------------------------
    def make_ticks_50(H, max_labels=6):
        step = 50
        while True:
            ticks = list(range(1, H + 1, step))
            if ticks[-1] != H:
                ticks.append(H)
            if len(ticks) <= max_labels or step >= H:
                break
            step += 50
        return [t - 1 for t in ticks], [str(t) for t in ticks]

    # -----------------------------
    # global color scales (separate for C and Q)
    # -----------------------------
    def compute_global_scale(value_getter, only_diagonal_Q=False):
        vals = []
        for (i, j, n), p in file_map.items():
            if i > j:
                continue
            if only_diagonal_Q and (i != j):
                continue
            d = load_npz(p)
            M = value_getter(d)
            vals.append(np.percentile(M, [vmin_percentile, vmax_percentile]))
        vals = np.vstack(vals)
        vmin = float(vals[:, 0].min())
        vmax = float(vals[:, 1].max())
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-6
        return vmin, vmax

    # -----------------------------
    # plot helpers
    # -----------------------------
    def plot_Q_row(fig, gs, row_index, vmin, vmax, show_titles=True):
        """
        Plot the Q-row:
          For each task j and epoch n, render Q_{jj}^{(n)} ordered by π_j.
          No i!=j Q is used.
        """
        last_im = None
        for j in range(J):
            epochs = epochs_per_task.get(j, [])
            if not epochs:
                continue
            col_start = task_col_start[j]
            #proto_order = proto_order_by_task.get(j, None)

            for e_idx, n in enumerate(epochs):
                ax = fig.add_subplot(gs[row_index, col_start + e_idx])

                key = (j, j, n)  # diagonal ONLY
                if key not in file_map:
                    ax.axis("off")
                    continue

                d = load_npz(file_map[key])
                if "Q" not in d:
                    ax.axis("off")
                    continue
                Q = d["Q"]
                row_order = sample_order_by_data.get(j, None)
                if row_order is not None and Q.shape[0] == len(row_order):
                    Q = Q[row_order]
                Q = Q[:, proto_order_global]
                

                last_im = ax.imshow(
                    Q,
                    aspect="auto",             # geometry already handled
                    interpolation="nearest",
                    vmin=vmin, vmax=vmax,
                )
                ax.set_anchor("NW")
                ax.set_xticks([])
                ax.set_yticks([])  # keep clean (you can add later if you want)

                if show_titles:
                    ax.set_title(f"#{n}", fontsize=5, pad=1.0)
                else:
                    ax.set_title("")

        return last_im

    def plot_C_triangle(fig, gs, row_offset, vmin, vmax, show_titles_first_row_only=True):
        """
        Plot C triangle rows i=0..J-1 into gs[row_offset+i, :].
        """
        last_im = None
        for i in range(J):
            row_order = sample_order_by_data.get(i, None)

            for j in range(i, J):
                epochs = epochs_per_task.get(j, [])
                if not epochs:
                    continue
                col_start = task_col_start[j]
                #proto_order = proto_order_by_task.get(j, None)

                for e_idx, n in enumerate(epochs):
                    ax = fig.add_subplot(gs[row_offset + i, col_start + e_idx])

                    key = (i, j, n)
                    if key not in file_map:
                        ax.axis("off")
                        continue

                    d = load_npz(file_map[key])
                    C = get_C_used(d)

                    C = C[:, proto_order_global]


                    last_im = ax.imshow(
                        C,
                        aspect="auto",
                        interpolation="nearest",
                        vmin=vmin, vmax=vmax,
                    )
                    ax.set_anchor("NW")
                    ax.set_xticks([])

                    # y ticks ONLY at data_i_task_i_epoch_first (diagonal, first epoch)
                    # if (j == i) and (n == first_epoch.get(i)):
                    #     yt, yl = make_ticks_50(C.shape[0])
                    #     ax.set_yticks(yt)
                    #     ax.set_yticklabels(yl, fontsize=7)
                    # else:
                    #     ax.set_yticks([])
                    ax.set_yticks([])
                    # titles only on the very top row (handled by Q-row), so keep C rows title-free
                    if show_titles_first_row_only:
                        ax.set_title("")
                    else:
                        ax.set_title(f"T{j}E{n}", fontsize=7, pad=1.0)

        return last_im

    def add_task_headers(fig, gs, fig_h, *, y_top_text=0.985, y_sub_text=0.965):
        for j in range(J):
            epochs = epochs_per_task.get(j, [])
            if not epochs:
                continue
            c0 = task_col_start[j]
            c1 = c0 + len(epochs) - 1
            ax0 = fig.add_subplot(gs[0, c0])
            ax1 = fig.add_subplot(gs[0, c1])
            p0, p1 = ax0.get_position(), ax1.get_position()
            ax0.remove()
            ax1.remove()
            x = 0.5 * (p0.x0 + p1.x1)
            fig.text(x, y_top_text, f"task {j}", ha="center", va="top", fontsize=9)
            #fig.text(x, y_sub_text, f"(#epochs={len(epochs)})",
            #         ha="center", va="top", fontsize=7, color="gray")

    # -----------------------------
    # Output paths
    # -----------------------------
    import os
    base, _ = os.path.splitext(out_png)
    out_C = base + "_C.png"
    out_Q = base + "_Q.png"
    os.makedirs(os.path.dirname(out_C) or ".", exist_ok=True)

    # -----------------------------
    # 1) C figure = Q-row + C-triangle
    # -----------------------------
    fig_w, fig_h = figsize_C
    fig = plt.figure(figsize=figsize_C, dpi=dpi)
    gs = fig.add_gridspec(
        nrows=total_rows_C,
        ncols=total_cols,
        left=left_margin / fig_w,
        right=1 - right_margin / fig_w,
        top=1 - top_margin / fig_h,
        bottom=bottom_margin / fig_h,
        wspace=0.01,
        hspace=row_spacing_in / subfig_h,
    )

    # scales
    vmin_C, vmax_C = compute_global_scale(lambda d: get_C_used(d), only_diagonal_Q=False)
    vmin_Q, vmax_Q = compute_global_scale(lambda d: d["Q"], only_diagonal_Q=True)

    # plot
    last_im_Q = plot_Q_row(fig, gs, row_index=0, vmin=vmin_Q, vmax=vmax_Q, show_titles=True)
    last_im_C = plot_C_triangle(fig, gs, row_offset=1, vmin=vmin_C, vmax=vmax_C, show_titles_first_row_only=True)

    add_task_headers(fig, gs, fig_h)

    # colorbars
    if last_im_C is not None:
        cbar = fig.colorbar(last_im_C, ax=fig.axes, fraction=0.012, pad=0.004)
        cbar.set_label("C = -logits", fontsize=9)
    # add temperature and epsilon info to the figure
    fig.text(0.99, 0.01, f"Temp={temperature_value}, Eps={epsilon_value}", 
                ha="right", va="bottom", fontsize=7, color="gray")
    fig.savefig(out_C, bbox_inches="tight")
    plt.close(fig)

    # # -----------------------------
    # # 2) Q-only figure (clean)
    # # -----------------------------
    # fig_w, fig_h = figsize_Q
    # fig = plt.figure(figsize=figsize_Q, dpi=dpi)
    # gs = fig.add_gridspec(
    #     nrows=total_rows_Q,
    #     ncols=total_cols,
    #     left=left_margin / fig_w,
    #     right=1 - right_margin / fig_w,
    #     top=1 - top_margin / fig_h,
    #     bottom=bottom_margin / fig_h,
    #     wspace=0.01,
    #     hspace=0.0,
    # )

    # last_im_Q = plot_Q_row(fig, gs, row_index=0, vmin=vmin_Q, vmax=vmax_Q, show_titles=True)
    # add_task_headers(fig, gs, fig_h)

    # if last_im_Q is not None:
    #     cbar = fig.colorbar(last_im_Q, ax=fig.axes, fraction=0.012, pad=0.004)
    #     cbar.set_label("Q (assignment, diagonal only)", fontsize=9)

    # fig.savefig(out_Q, bbox_inches="tight")
    # plt.close(fig)

    return out_C#, out_Q


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
     
    # read log_dir from bash argument
    import argparse
    parser = argparse.ArgumentParser(description="Plot cost triangle from logged data.")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing logged data files.")
    args = parser.parse_args()
    log_dir = args.log_dir
    

    out = plot_cost_triangle(
        log_dir=log_dir,
        out_png=log_dir+"/cost_triangle2.png",
    )
    print("Saved:", out)
