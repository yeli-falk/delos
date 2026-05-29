#!/usr/bin/env python3
"""
Parse a training .out log and plot losses.

Recognises two kinds of lines:

  [step   350] loss=0.0392 mh=0.0167 fut=0.0225 (2541.4s)
  train loss=0.0380  val loss=0.0169  (mh=0.0039 fut=0.0130)

Per-epoch markers '[epoch N/M]' are used to assign each step its epoch.

Usage:
    python plot_losses.py path/to/run.out
    python plot_losses.py path/to/run.out -o losses.png
    python plot_losses.py path/to/run.out --no-steps    # epoch-level only
    python plot_losses.py path/to/run.out --log         # log-scale y axis
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt


# ---------- regex ----------------------------------------------------------

EPOCH_RE = re.compile(r"\[epoch\s+(\d+)\s*/\s*(\d+)\]")

STEP_RE = re.compile(
    r"\[step\s+(\d+)\]\s+"
    r"loss=([0-9.eE+-]+)\s+"
    r"mh=([0-9.eE+-]+)\s+"
    r"fut=([0-9.eE+-]+)"
)

EPOCH_SUMMARY_RE = re.compile(
    r"train loss=([0-9.eE+-]+)\s+"
    r"val loss=([0-9.eE+-]+)\s+"
    r"\(mh=([0-9.eE+-]+)\s+fut=([0-9.eE+-]+)\)"
)


# ---------- data containers -----------------------------------------------

@dataclass
class Run:
    # step-level (one entry per logged step, across all epochs)
    step_epoch: list[int] = field(default_factory=list)   # which epoch the step belongs to
    step_idx:   list[int] = field(default_factory=list)   # the 'step N' value reported
    step_loss:  list[float] = field(default_factory=list)
    step_mh:    list[float] = field(default_factory=list)
    step_fut:   list[float] = field(default_factory=list)

    # epoch-level summaries
    ep_num:    list[int] = field(default_factory=list)
    ep_train:  list[float] = field(default_factory=list)
    ep_val:    list[float] = field(default_factory=list)
    ep_val_mh: list[float] = field(default_factory=list)
    ep_val_fut:list[float] = field(default_factory=list)


# ---------- parsing -------------------------------------------------------

def parse_log(path: Path) -> Run:
    run = Run()
    current_epoch: int | None = None
    steps_per_epoch_seen: list[int] = []  # how many step lines we saw in each epoch

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                current_epoch = int(m.group(1))
                steps_per_epoch_seen.append(0)
                continue

            m = STEP_RE.search(line)
            if m and current_epoch is not None:
                run.step_epoch.append(current_epoch)
                run.step_idx.append(int(m.group(1)))
                run.step_loss.append(float(m.group(2)))
                run.step_mh.append(float(m.group(3)))
                run.step_fut.append(float(m.group(4)))
                steps_per_epoch_seen[-1] += 1
                continue

            m = EPOCH_SUMMARY_RE.search(line)
            if m and current_epoch is not None:
                run.ep_num.append(current_epoch)
                run.ep_train.append(float(m.group(1)))
                run.ep_val.append(float(m.group(2)))
                run.ep_val_mh.append(float(m.group(3)))
                run.ep_val_fut.append(float(m.group(4)))

    return run


def fractional_epoch(run: Run) -> list[float]:
    """
    Map every step to a fractional epoch number, so we can plot step-level
    curves on the same x-axis as epoch summaries.

    We assume the steps logged within one epoch are evenly spaced across it.
    """
    # count how many step lines per epoch
    per_epoch: dict[int, int] = {}
    for e in run.step_epoch:
        per_epoch[e] = per_epoch.get(e, 0) + 1

    seen: dict[int, int] = {e: 0 for e in per_epoch}
    out: list[float] = []
    for e in run.step_epoch:
        seen[e] += 1
        # place the k-th of N step lines at e-1 + k/N
        out.append((e - 1) + seen[e] / per_epoch[e])
    return out


# ---------- plotting ------------------------------------------------------

def plot(run: Run, out_path: Path, show_steps: bool, log_y: bool) -> None:
    if not run.ep_num and not run.step_loss:
        raise SystemExit("No loss lines found in the log.")

    fig, ax = plt.subplots(figsize=(11, 6))

    if show_steps and run.step_loss:
        x = fractional_epoch(run)
        ax.plot(x, run.step_loss, color="#9ecae1", linewidth=0.8,
                label="train (per step)", zorder=1)
        ax.plot(x, run.step_mh, color="#a1d99b", linewidth=0.6,
                linestyle="--", label="train mh (per step)", zorder=1)
        ax.plot(x, run.step_fut, color="#bcbddc", linewidth=0.6,
                linestyle="--", label="train fut (per step)", zorder=1)

    if run.ep_num:
        ax.plot(run.ep_num, run.ep_train, "o-", color="#1f77b4",
                linewidth=2, markersize=5, label="train (epoch)", zorder=3)
        ax.plot(run.ep_num, run.ep_val, "s-", color="#d62728",
                linewidth=2, markersize=5, label="val", zorder=3)
        ax.plot(run.ep_num, run.ep_val_mh, "^--", color="#2ca02c",
                linewidth=1.2, markersize=4, label="val mh", zorder=2)
        ax.plot(run.ep_num, run.ep_val_fut, "v--", color="#9467bd",
                linewidth=1.2, markersize=4, label="val fut", zorder=2)

        # mark best val
        best_i = min(range(len(run.ep_val)), key=lambda i: run.ep_val[i])
        bx, by = run.ep_num[best_i], run.ep_val[best_i]
        ax.scatter([bx], [by], s=120, facecolors="none",
                   edgecolors="#d62728", linewidths=2, zorder=4)
        ax.annotate(f"best val={by:.4f} @ epoch {bx}",
                    xy=(bx, by), xytext=(8, -14), textcoords="offset points",
                    fontsize=9, color="#d62728")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training curves — {out_path.stem}")
    ax.grid(True, alpha=0.3)
    if log_y:
        ax.set_yscale("log")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


# ---------- cli -----------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logfile", type=Path, help="Path to the training .out file")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output image path (default: <logfile>.png)")
    p.add_argument("--no-steps", action="store_true",
                   help="Only plot per-epoch summaries, hide step-level curves")
    p.add_argument("--log", action="store_true",
                   help="Use a log-scale y axis")
    args = p.parse_args()

    if not args.logfile.exists():
        raise SystemExit(f"file not found: {args.logfile}")

    out_path = args.output or args.logfile.with_suffix(".png")

    run = parse_log(args.logfile)
    print(f"parsed {len(run.ep_num)} epochs, {len(run.step_loss)} step lines")
    plot(run, out_path, show_steps=not args.no_steps, log_y=args.log)


if __name__ == "__main__":
    main()