"""Visualise training loss curves from a SLURM log file.

Parses the `[epoch N/M]` / `train loss=... val loss=...` / `[step ...]`
lines we already emit and produces a matplotlib plot. Handles both
METR-LA-style logs (with `mh=`/`fut=` breakdown) and DBLP-style logs
(overall loss only).

Usage:
    python scripts/plot_loss.py logs/dblp-cjepa-1234567.out
    python scripts/plot_loss.py logs/metrla-cjepa-987.out --out fig.pdf
    python scripts/plot_loss.py logs/*.out                # multi-run overlay
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Dict


# JEPA training epoch summary:
#   "  train loss=0.7068  val loss=0.3545"
#   "  train loss=0.0240  val loss=0.0107  (mh=0.0021 fut=0.0086)"
RE_EPOCH = re.compile(
    r"train loss=([\d.eE+-]+)\s+val loss=([\d.eE+-]+)"
    r"(?:\s+\(mh=([\d.eE+-]+)\s+fut=([\d.eE+-]+)\))?"
)

# Edge-masking pretraining epoch summary:
#   "  train: bce=0.6929 acc=0.532  val: bce=0.6826 acc=0.501"
RE_EPOCH_PRETRAIN = re.compile(
    r"train:\s+bce=([\d.eE+-]+)\s+acc=([\d.eE+-]+)"
    r"\s+val:\s+bce=([\d.eE+-]+)\s+acc=([\d.eE+-]+)"
)

# Feature-reconstruction pretraining epoch summary (MSE only, no acc):
#   "  train loss=1.1698  val loss=1.0752"
# Matches the same RE_EPOCH pattern; no separate regex needed.

# Probe-fitting line:
#   "  [probe epoch 1] bce=0.000018"
#   "  [probe epoch 1] mse=2.2502"
RE_PROBE_EPOCH = re.compile(
    r"\[probe epoch (\d+)\]\s+(?:bce|mse)=([\d.eE+-]+)"
)

# Per-step line:
#   "  [step    50] loss=0.1803 mh=0.0894 fut=0.0909 (194.4s)"
#   "  [step    50] loss=0.5116 mh=0.2557 fut=0.2559 lr=6.82e-05 (363.5s)"
RE_STEP = re.compile(
    r"\[step\s+(\d+)\]\s+loss=([\d.eE+-]+)"
    r"(?:\s+mh=([\d.eE+-]+))?"
    r"(?:\s+fut=([\d.eE+-]+))?"
    r"(?:\s+lr=([\d.eE+-]+))?"
)

# Test-metric lines:
#   "  + 2 steps   8.012  11.546  0.212"      (METR-LA MAE/RMSE/MAPE)
#   "  + 1 year: MRR=0.1846 Hits@1=0.1072 ..."  (DBLP)
RE_TEST_METRLA = re.compile(
    r"\+\s*(\d+)\s+steps\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)
RE_TEST_DBLP = re.compile(
    r"\+\s*(\d+)\s+year:\s+MRR=([\d.]+)\s+Hits@1=([\d.]+)\s+Hits@3=([\d.]+)\s+Hits@10=([\d.]+)"
)


def parse_log(path: Path) -> Dict:
    epochs: List[Dict] = []
    steps: List[Dict] = []
    tests: List[Dict] = []
    probe: List[Dict] = []
    for line in path.read_text().splitlines():
        if (m := RE_EPOCH_PRETRAIN.search(line)):
            epochs.append({
                "train": float(m.group(1)),
                "train_acc": float(m.group(2)),
                "val": float(m.group(3)),
                "val_acc": float(m.group(4)),
            })
            continue
        if (m := RE_EPOCH.search(line)):
            e = {"train": float(m.group(1)), "val": float(m.group(2))}
            if m.group(3) is not None:
                e["val_mh"] = float(m.group(3))
                e["val_fut"] = float(m.group(4))
            epochs.append(e)
            continue
        if (m := RE_PROBE_EPOCH.search(line)):
            probe.append({"epoch": int(m.group(1)),
                          "loss": float(m.group(2))})
            continue
        if (m := RE_STEP.search(line)):
            s = {"step_in_epoch": int(m.group(1)),
                 "loss": float(m.group(2))}
            if m.group(3) is not None:
                s["mh"] = float(m.group(3))
            if m.group(4) is not None:
                s["fut"] = float(m.group(4))
            if m.group(5) is not None:
                s["lr"] = float(m.group(5))
            steps.append(s)
            continue
        if (m := RE_TEST_METRLA.search(line)):
            tests.append({"horizon": int(m.group(1)),
                          "MAE": float(m.group(2)),
                          "RMSE": float(m.group(3)),
                          "MAPE": float(m.group(4))})
            continue
        if (m := RE_TEST_DBLP.search(line)):
            tests.append({"horizon": int(m.group(1)),
                          "MRR": float(m.group(2)),
                          "Hits@1": float(m.group(3)),
                          "Hits@3": float(m.group(4)),
                          "Hits@10": float(m.group(5))})
    return {"epochs": epochs, "steps": steps, "tests": tests,
            "probe": probe, "path": path}


def plot_single(run: Dict, ax) -> None:
    epochs = run["epochs"]
    if not epochs:
        ax.text(0.5, 0.5, f"no epoch summaries in\n{run['path'].name}",
                ha="center", va="center", transform=ax.transAxes)
        return

    x = list(range(1, len(epochs) + 1))
    train = [e["train"] for e in epochs]
    val = [e["val"] for e in epochs]

    ax.plot(x, train, marker="o", markersize=4, linewidth=1.6,
            color="C0", label="train")
    ax.plot(x, val, marker="s", markersize=4, linewidth=1.6,
            color="C3", label="val")
    if "val_mh" in epochs[0]:
        ax.plot(x, [e["val_mh"] for e in epochs], linestyle="--",
                color="C2", linewidth=1.2, marker="^", markersize=3,
                label="val mh")
        ax.plot(x, [e["val_fut"] for e in epochs], linestyle="--",
                color="C4", linewidth=1.2, marker="v", markersize=3,
                label="val fut")

    # Best-val annotation.
    best_idx = min(range(len(val)), key=lambda i: val[i])
    ax.scatter([x[best_idx]], [val[best_idx]],
               s=180, facecolors="none", edgecolors="red", linewidths=2,
               zorder=5)
    ax.annotate(f"best val={val[best_idx]:.4f} @ epoch {x[best_idx]}",
                xy=(x[best_idx], val[best_idx]),
                xytext=(8, -14), textcoords="offset points",
                color="red", fontsize=9)

    # Per-step background trace (lighter).
    steps = run["steps"]
    if steps and len(steps) > len(epochs):
        # Reconstruct a "global step" axis by interpolation across epochs.
        # We just plot the step index normalised to [1, num_epochs].
        n_steps = len(steps)
        x_step = [1 + (i / max(n_steps - 1, 1)) * (len(epochs) - 1)
                  for i in range(n_steps)]
        loss_step = [s["loss"] for s in steps]
        ax.plot(x_step, loss_step, color="C0", alpha=0.25, linewidth=0.8,
                label="train (per step)")
        if "mh" in steps[0]:
            ax.plot(x_step, [s["mh"] for s in steps],
                    color="C2", alpha=0.25, linewidth=0.8, linestyle="--",
                    label="train mh (per step)")
            ax.plot(x_step, [s["fut"] for s in steps],
                    color="C4", alpha=0.25, linewidth=0.8, linestyle="--",
                    label="train fut (per step)")

    # Pretraining-style logs also carry accuracy; draw it on a secondary axis.
    has_acc = "train_acc" in epochs[0]
    if has_acc:
        ax2 = ax.twinx()
        ax2.plot(x, [e["train_acc"] for e in epochs],
                 color="C0", alpha=0.5, linestyle=":", marker="o",
                 markersize=3, label="train acc")
        ax2.plot(x, [e["val_acc"] for e in epochs],
                 color="C3", alpha=0.5, linestyle=":", marker="s",
                 markersize=3, label="val acc")
        ax2.set_ylabel("Accuracy", color="grey")
        ax2.set_ylim(0.45, 1.0)
        ax2.tick_params(axis="y", labelcolor="grey")
        ax2.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # Probe loss as a small overlaid trace at the end of the x-axis.
    if run.get("probe"):
        n_ep = len(epochs)
        probe_x = [n_ep + 0.3 + 0.5 * i for i in range(len(run["probe"]))]
        probe_y = [p["loss"] for p in run["probe"]]
        ax.plot(probe_x, probe_y, color="C5", marker="x", linewidth=1.2,
                label="probe")

    ax.set_xlabel("Epoch")
    is_pretrain = has_acc
    ax.set_ylabel("BCE loss" if is_pretrain else "Loss")
    if not is_pretrain:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    label_kind = "pretrain" if is_pretrain else "train"
    ax.set_title(f"{label_kind} curves — {run['path'].stem}", fontsize=11)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)


def plot_runs(runs: List[Dict], out: Path) -> None:
    import matplotlib.pyplot as plt
    n = len(runs)
    if n == 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_single(runs[0], ax)
    else:
        cols = min(2, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.2 * rows),
                                 squeeze=False)
        for run, ax in zip(runs, axes.flatten()):
            plot_single(run, ax)
        for ax in axes.flatten()[n:]:
            ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] -> {out}")


def print_test_table(runs: List[Dict]) -> None:
    for run in runs:
        if not run["tests"]:
            continue
        print(f"\n[test] {run['path'].name}")
        keys = [k for k in run["tests"][0].keys() if k != "horizon"]
        header = f"  horizon  " + "  ".join(f"{k:>8}" for k in keys)
        print(header)
        for t in run["tests"]:
            row = f"  +{t['horizon']:>5}  " + "  ".join(
                f"{t[k]:8.4f}" for k in keys
            )
            print(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", help="One or more SLURM log files.")
    p.add_argument("--out", default=None,
                   help="Output PNG/PDF (default: <first-log>.png)")
    args = p.parse_args()

    runs = [parse_log(Path(p)) for p in args.logs]
    if args.out:
        out = Path(args.out)
    elif len(runs) == 1:
        out = Path(runs[0]["path"]).with_suffix(".png")
    else:
        out = Path("loss_curves.png")
    plot_runs(runs, out)
    print_test_table(runs)


if __name__ == "__main__":
    main()
