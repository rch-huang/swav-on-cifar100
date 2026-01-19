import numpy as np
import matplotlib.pyplot as plt
import json
import os

def replay_from_npz(bundle_npz: str, out_png: str, meta_json: str = None, compare_png: str = None):
    d = np.load(bundle_npz, allow_pickle=True)

    C_used_raw = d["C_used_raw"]      # [B,K]
    Q_raw      = d["Q_raw"]           # [B,K]
    row_idx    = d["row_idx"].astype(np.int64)
    col_idx    = d["col_idx"].astype(np.int64)

    # recompute "sorted" strictly by applying indices
    C_used_re = C_used_raw[row_idx][:, col_idx]
    Q_re      = Q_raw[row_idx][:, col_idx]

    # compare to stored sorted (ground truth for the first run)
    C_used_sorted_saved = d["C_used_sorted"]
    Q_sorted_saved      = d["Q_sorted"]

    def _max_abs(a,b): return float(np.max(np.abs(a-b)))
    print("[Replay] max|C_sorted - C_re| =", _max_abs(C_used_sorted_saved, C_used_re))
    print("[Replay] max|Q_sorted - Q_re| =", _max_abs(Q_sorted_saved, Q_re))

    # Load vmin/vmax from meta if provided, else compute from percentiles again
    C_vmin = C_vmax = Q_vmin = Q_vmax = None
    if meta_json is not None and os.path.exists(meta_json):
        meta = json.load(open(meta_json, "r"))
        C_vmin, C_vmax = meta.get("C_vmin"), meta.get("C_vmax")
        Q_vmin, Q_vmax = meta.get("Q_vmin"), meta.get("Q_vmax")

    # plot with the same scales => should visually match
    fig, axes = plt.subplots(1,2, figsize=(14,6.2))
    ax0, ax1 = axes
    im0 = ax0.imshow(C_used_re, aspect="auto", interpolation="nearest", vmin=C_vmin, vmax=C_vmax)
    ax0.set_title("Replayed Cost matrix C_used (sorted)")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    im1 = ax1.imshow(Q_re, aspect="auto", interpolation="nearest", vmin=Q_vmin, vmax=Q_vmax)
    ax1.set_title("Replayed Assignment matrix Q (sorted)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)
    print("[Replay] saved:", out_png)

    # Optional pixel-level comparison (only if you provide original png)
    if compare_png is not None:
        from PIL import Image
        a = np.array(Image.open(out_png).convert("RGB"), dtype=np.int16)
        b = np.array(Image.open(compare_png).convert("RGB"), dtype=np.int16)
        if a.shape != b.shape:
            print("[Replay] PNG shapes differ:", a.shape, b.shape, "(cannot pixel-compare)")
        else:
            diff = np.abs(a-b)
            print("[Replay] PNG mean abs diff:", float(diff.mean()), "max abs diff:", int(diff.max()))


if __name__ == "__main__":
    replay_from_npz(
        bundle_npz="log_20260114124038_0_60/debug_bundle.npz",
        out_png="log_20260114124038_0_60/recomputed_sorted_visualization.png",
        meta_json="log_20260114124038_0_60/debug_bundle_meta.json",  
        compare_png="log_20260114005854/cost_data0_onTask0_epoch60.png.png"
    )