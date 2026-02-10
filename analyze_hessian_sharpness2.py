#!/usr/bin/env python3
"""
ABSOLUTE COVERAGE VERSION (FINAL)

Key semantic guarantee:
- Energy ratios are ALWAYS normalized by ||Δw||^2 (full energy).
- Top-K coverage may be < 1.0 → empty region at the top is preserved.
- NO top-K renormalization.
- NO artificial filling to 100%.

Plots:
1) Anchor-level energy spectrum (mean + 10–90% band)
2) Overall sharpness curves
3) Overall cumulative coverage heatmap (absolute, with blank top region)
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ---------------- regex ----------------

TASK_RE   = re.compile(r".*task_?(\d+)$")
ANCHOR_RE = re.compile(r"anchor_(\d+)$")
EPOCH_RE  = re.compile(r"epoch_(\d+)$")
STEP_RE   = re.compile(r"step_(\d+)$")

# ---------------- utils ----------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sorted_dirs(parent: Path, regex):
    out = []
    for d in parent.iterdir():
        if d.is_dir():
            m = regex.match(d.name)
            if m:
                out.append((int(m.group(1)), d))
    return sorted(out, key=lambda x: x[0])

def load_energy_ratio(npz_path: Path):
    """
    Return absolute energy ratios:
      e_i = <Δw, u_i>^2 / ||Δw||^2
    Sum(e) <= 1 in general.
    """
    if not npz_path.is_file():
        return None
    try:
        d = np.load(npz_path)
        if "energy" not in d:
            return None
        e = d["energy"].reshape(-1)
        return e / (e.sum() + 1e-12)
    except Exception:
        return None

def load_eigvals(anchor_dir: Path, block: str):
    p = anchor_dir / f"{block}_eigvals.json"
    if not p.is_file():
        return None
    with p.open("r") as f:
        j = json.load(f)
    return np.asarray(j["eigvals"], dtype=np.float64)

# ---------------- global y-lim for spectrum ----------------

def compute_global_energy_ylim(root: Path, block: str, K: int):
    ymax = 0.0
    for _, task_dir in sorted_dirs(root, TASK_RE):
        for _, anchor_dir in sorted_dirs(task_dir, ANCHOR_RE):
            for _, epoch_dir in sorted_dirs(anchor_dir, EPOCH_RE):
                for _, step_dir in sorted_dirs(epoch_dir, STEP_RE):
                    e = load_energy_ratio(step_dir / f"{block}_raw.npz")
                    if e is None:
                        continue
                    ymax = max(ymax, float(np.max(e[:K])))
    return ymax * 1.05

# ---------------- anchor energy spectrum ----------------

def plot_anchor_energy_spectrum(
    task_dir: Path,
    anchor_id: int,
    block: str,
    K: int,
    global_ylim: float,
    outdir: Path,
):
    anchor_dir = task_dir / f"anchor_{anchor_id}"
    spectra = []

    for _, epoch_dir in sorted_dirs(anchor_dir, EPOCH_RE):
        for _, step_dir in sorted_dirs(epoch_dir, STEP_RE):
            e = load_energy_ratio(step_dir / f"{block}_raw.npz")
            if e is None or len(e) < K:
                continue
            spectra.append(e[:K])

    if not spectra:
        return

    E = np.stack(spectra, axis=0)
    mean = E.mean(axis=0)
    lo   = np.percentile(E, 10, axis=0)
    hi   = np.percentile(E, 90, axis=0)

    x = np.arange(1, K + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, mean, lw=2, label="mean")
    ax.fill_between(x, lo, hi, alpha=0.3, label="10–90%")

    ax.set_xlim(1, K)
    ax.set_ylim(0.0, global_ylim)
    ax.set_xlabel("eigen-direction index (sorted by curvature)")
    ax.set_ylabel("energy ratio (absolute)")
    ax.set_title(f"{task_dir.name} | anchor={anchor_id} | {block}")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(outdir / f"{block}_anchor_{anchor_id:04d}_energy_spectrum.png")
    plt.close(fig)

# ---------------- overall sharpness ----------------

def plot_overall_sharpness(root: Path, block: str, K: int, outdir: Path):
    xs, y1, yk, anchor_labels = [], [], [], []
    major_x, major_lbl = [], []
    cursor = 0

    for t_id, task_dir in sorted_dirs(root, TASK_RE):
        if t_id >0:
            continue  # DEBUG: only first task
        major_x.append(cursor)
        major_lbl.append(f"T_{t_id}")

        for anchor_id, anchor_dir in sorted_dirs(task_dir, ANCHOR_RE):
            ev = load_eigvals(anchor_dir, block)
            if ev is None:
                continue
            k_eff = min(K, len(ev))
            xs.append(cursor)
            y1.append(float(ev[0]))
            yk.append(float(ev[:k_eff].sum()))
            anchor_labels.append(str(anchor_id))
            cursor += 1

    if not xs:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, y1, marker="o", lw=1.2, label="top-1")
    ax.plot(xs, yk, marker="s", lw=1.2, label=f"top-{K}")

    ax.set_xticks(major_x)
    ax.set_xticklabels(major_lbl)
    ax.set_xticks(xs, minor=True)
    ax.set_xticklabels(anchor_labels, minor=True)
    ax.tick_params(axis="x", which="minor", rotation=90, labelsize=8)

    for p in major_x[1:]:
        ax.axvline(p - 0.5, color="gray", linestyle="--", alpha=0.5)

    ax.set_title(f"Overall sharpness | {block}")
    ax.set_xlabel("anchors (concatenated across tasks)")
    ax.set_ylabel("eigenvalue")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(outdir / f"overall_{block}_sharpness.png")
    plt.close(fig)

# ---------------- overall absolute coverage heatmap ----------------

def _anchor_mean_topk_energy(anchor_dir: Path, block: str, K: int):
    """
    Mean absolute top-K energy over steps.
    No renormalization: sum <= 1.
    """
    vecs = []
    for _, epoch_dir in sorted_dirs(anchor_dir, EPOCH_RE):
        for _, step_dir in sorted_dirs(epoch_dir, STEP_RE):
            e = load_energy_ratio(step_dir / f"{block}_raw.npz")
            if e is None or len(e) < K:
                continue
            vecs.append(e[:K])

    if not vecs:
        return None

    return np.mean(np.stack(vecs, axis=0), axis=0)
def smooth_along_anchors(ecols, kernel=(1, 2, 1)):
    """
    ecols: list of [K] vectors, one per anchor
    Smooth ONLY along anchor dimension.
    """
    W = np.array(kernel, dtype=np.float32)
    W = W / W.sum()

    E = np.stack(ecols, axis=0)  # [A, K]
    A, K = E.shape
    E_smooth = np.zeros_like(E)

    for a in range(A):
        for offset, w in zip(range(-1, 2), W):
            aa = a + offset
            if 0 <= aa < A:
                E_smooth[a] += w * E[aa]

    return E_smooth

def plot_overall_coverage_heatmap(
    root: Path,
    block: str,
    K: int,
    outdir: Path,
    y_res: int = 400,
):
    xs = []
    anchor_labels = []
    major_x, major_lbl = [], []
    ecols = []
    cursor = 0

    for t_id, task_dir in sorted_dirs(root, TASK_RE):
        if t_id >0:
            continue  # DEBUG: only first task
        major_x.append(cursor)
        major_lbl.append(f"T_{t_id}")

        for anchor_id, anchor_dir in sorted_dirs(task_dir, ANCHOR_RE):
            e_bar = _anchor_mean_topk_energy(anchor_dir, block, K)
            if e_bar is None:
                continue
            xs.append(cursor)
            anchor_labels.append(str(anchor_id))
            ecols.append(e_bar)
            cursor += 1

    if not ecols:
        return
    ecols = smooth_along_anchors(ecols)

    n = len(ecols)
    img = np.zeros((y_res, n), dtype=np.float32)

    for j, e_bar in enumerate(ecols):
        cum = 0.0
        for i in range(K):
            share = float(e_bar[i])
            if share <= 0:
                continue
            y0 = int(np.floor(cum * y_res))
            cum += share
            y1 = int(np.floor(cum * y_res))
            y1 = min(y1, y_res)
            if y1 > y0:
                img[y0:y1, j] = i + 1
        # NOTE: no filling beyond cum → blank region preserved

    cmap = plt.get_cmap("YlGn")
    norm = Normalize(vmin=1, vmax=K)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(
        img,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    yticks = [0, int(0.25*y_res), int(0.5*y_res), int(0.75*y_res), y_res-1]
    ax.set_yticks(yticks)
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

    ax.set_xticks(major_x)
    ax.set_xticklabels(major_lbl)
    ax.set_xticks(xs, minor=True)
    ax.set_xticklabels(anchor_labels, minor=True)
    ax.tick_params(axis="x", which="minor", rotation=90, labelsize=8)

    for p in major_x[1:]:
        ax.axvline(p - 0.5, color="gray", linestyle="--", alpha=0.6)

    ax.set_xlabel("anchors (concatenated across tasks)")
    ax.set_ylabel("absolute cumulative coverage")
    ax.set_title(f"Overall top-{K} absolute coverage heatmap | {block}")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ticks = [1, 2, 5, 10, 20, K] if K >= 20 else list(range(1, K + 1))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, ticks=ticks)
    cbar.set_label("eig index (top1 → topK)")

    fig.tight_layout()
    fig.savefig(outdir / f"overall_{block}_absolute_coverage_heatmap_top{K}.png")
    plt.close(fig)

# ---------------- main ----------------

def main(
    root_dir: str,
    theta_K: int,
    C_K: int,
    heatmap_K_theta: int,
    heatmap_K_C: int,
    y_res: int,
):
    root = Path(root_dir).resolve()

    theta_ylim = compute_global_energy_ylim(root, "theta", theta_K)
    C_ylim     = compute_global_energy_ylim(root, "C", C_K)

    for _, task_dir in sorted_dirs(root, TASK_RE):
        if _ >0:
            continue  # DEBUG: only first task
        outdir = task_dir / "analysis_plots"
        ensure_dir(outdir)
        for anchor_id, _ in sorted_dirs(task_dir, ANCHOR_RE):
            plot_anchor_energy_spectrum(task_dir, anchor_id, "theta", theta_K, theta_ylim, outdir)
            plot_anchor_energy_spectrum(task_dir, anchor_id, "C", C_K, C_ylim, outdir)

    overall_out = root / "analysis_plots"
    ensure_dir(overall_out)

    plot_overall_sharpness(root, "theta", theta_K, overall_out)
    plot_overall_sharpness(root, "C", C_K, overall_out)

    plot_overall_coverage_heatmap(root, "theta", heatmap_K_theta, overall_out, y_res=y_res)
    plot_overall_coverage_heatmap(root, "C", heatmap_K_C, overall_out, y_res=y_res)

    print("Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root_dir")
    ap.add_argument("--theta_K", type=int, default=50)
    ap.add_argument("--C_K", type=int, default=5)
    ap.add_argument("--heatmap_K_theta", type=int, default=10)
    ap.add_argument("--heatmap_K_C", type=int, default=300)
    ap.add_argument("--y_res", type=int, default=400)
    args = ap.parse_args()

    main(
        args.root_dir,
        theta_K=args.theta_K,
        C_K=args.C_K,
        heatmap_K_theta=args.heatmap_K_theta,
        heatmap_K_C=args.heatmap_K_C,
        y_res=args.y_res,
    )
