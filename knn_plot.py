import re
import sys
import os
import matplotlib.pyplot as plt
from collections import defaultdict


def plot_knn_from_log(txt_path):
    # ---------- 1. Read file ----------
    lines = []
    with open(txt_path, "r") as f:
        # grep line with 'KNN eval on Task'
        line = f.readline()
        while line:
            if "KNN eval on Task" in line:
                lines.append(line.strip())
            if "Starting training on task " in line:
                lines.append("Eval at Task " + line.split("Starting training on task ")[-1].strip())
            line = f.readline()
            
    print(f"[Info] Found {len(lines)} KNN eval lines as follow:")
    for l in lines:
        print("  ", l) 

    # ---------- 2. Parse records ----------
    records = []
    current_task = None

    for line in lines:
        if line.startswith("Eval at Task"):
            current_task = int(re.findall(r"\d+", line)[0])

        elif "KNN eval on Task" in line:
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            epoch = int(nums[0])
            eval_task = int(nums[1])
            acc = float(nums[2])
            records.append((current_task, epoch, eval_task, acc))

    if not records:
        raise RuntimeError("No valid KNN records found.")

    # ---------- 3. Organize curves ----------
    curves = defaultdict(list)
    for task, epoch, eval_task, acc in records:
        global_epoch = task * 120 + epoch
        curves[eval_task].append((global_epoch, acc))

    for k in curves:
        curves[k] = sorted(curves[k], key=lambda x: x[0])

    # ---------- 4. Plot curves ----------
    plt.figure()
    final_infos = []  # (final_acc, eval_task, color)

    for eval_task, pts in curves.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        line, = plt.plot(xs, ys, label=f"Task {eval_task}")
        color = line.get_color()

  
        plt.scatter(xs[0], ys[0], s=20, marker="o", color=color, zorder=3)

        # 记录末尾信息
        final_infos.append((ys[-1], eval_task, color))

    # ---------- 5. Right-side sorted annotations ----------
    final_infos.sort(key=lambda x: x[0], reverse=True)

    ymin, ymax = 0, 100
    plt.ylim(ymin, ymax)

    n = len(final_infos)
    y_positions = list(
        reversed(
            [ymin + (i + 1) * (ymax - ymin) / (n + 1) for i in range(n)]
        )
    )

    x_text = max(x for pts in curves.values() for x, _ in pts) + 12

    for (acc, task_id, color), y in zip(final_infos, y_positions):
        plt.text(
            x_text,
            y,
            f"{acc:.1f} (#{task_id})",
            color=color,
            fontsize=9,
            va="center"
        )

    # ---------- 6. Axes ----------
    plt.xlabel("Global Epoch")
    plt.ylabel("KNN Accuracy (%)")

 
    plt.legend(
        loc="upper left",
        fontsize=8,
        frameon=True,
        handlelength=1.5,
        borderpad=0.4,
        labelspacing=0.3
    )

    plt.tight_layout()

    # ---------- 7. Save ----------
    png_path = os.path.splitext(txt_path)[0] + ".png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {png_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_knn_curves.py <knn_log.txt>")
        sys.exit(1)

    plot_knn_from_log(sys.argv[1])
