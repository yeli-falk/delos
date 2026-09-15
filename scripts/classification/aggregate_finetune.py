"""Aggregate the fine-tuning / label-efficiency cells into one table.

Reads every ``finetune.json`` written by ``scripts/classification/slurm_finetune.sh``
under ``<root>/<init>_aux<w>_f<frac>_s<seed>/`` and prints, per (init, aux)
row and label-fraction column, the mean ± std over seeds of one test metric.

The question the table answers is "does the initialisation buy anything once
the whole network is trained on labels?", so read it column-wise at the LOW
fractions: that is where DyG2Vec-style label efficiency shows. At 100% the
rows usually merge, and that is fine.

Usage:
    python scripts/classification/aggregate_finetune.py \
        --root outputs/cls_mooc_full/finetune --seeds 42 43 44
    python scripts/classification/aggregate_finetune.py \
        --root outputs/cls_mooc_full/finetune --metric onset_auroc --latex
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

PAT = re.compile(r"^(?P<init>[a-z_]+)_aux(?P<aux>[0-9.e-]+)_f(?P<frac>[0-9.e-]+)_s(?P<seed>\d+)$")

# Display order for rows.
INIT_ORDER = ["random", "encoder", "jepa", "ddgcl", "cldg", "idol"]
INIT_LABEL = {"random": "random init", "encoder": "edge-mask enc.",
              "jepa": "DyG-JEPA (enc.+pred.)", "ddgcl": "DDGCL enc.",
              "cldg": "CLDG enc.", "idol": "IDOL enc."}


def metric_of(blob: dict, metric: str, split: str):
    m = blob.get(split, {})
    if metric.startswith("onset_"):
        m = m.get("subset_onset", {})
        metric = metric[len("onset_"):]
    v = m.get(metric)
    return None if v is None or v != v else float(v)     # drop NaN


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0, len(vals))


def fmt(cell, prec, latex=False, best=False):
    if cell is None:
        return "--"
    mu, sd, n = cell
    if latex:
        body = f"{mu:.{prec}f} \\pm {sd:.{prec}f}" if n > 1 else f"{mu:.{prec}f}"
        return f"$\\mathbf{{{body}}}$" if best else f"${body}$"
    s = f"{mu:.{prec}f} ± {sd:.{prec}f}" if n > 1 else f"{mu:.{prec}f}"
    return s + ("*" if best else "")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="e.g. outputs/cls_mooc_full/finetune")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--metric", default="auroc",
                   help="auroc | auprc | onset_auroc | onset_auprc")
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--precision", type=int, default=3)
    p.add_argument("--latex", action="store_true")
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"{root} does not exist")
    cells = {}          # (init, aux, frac) -> {seed: blob}
    for d in sorted(root.iterdir()):
        m = PAT.match(d.name)
        f = d / "finetune.json"
        if not m or not f.exists():
            continue
        with open(f) as fh:
            blob = json.load(fh)
        if blob.get("smoke"):
            continue
        key = (m["init"], float(m["aux"]), float(m["frac"]))
        cells.setdefault(key, {})[int(m["seed"])] = blob
    if not cells:
        raise SystemExit(f"no finetune.json under {root}")

    fracs = sorted({k[2] for k in cells})
    rows = sorted({(k[0], k[1]) for k in cells},
                  key=lambda r: (INIT_ORDER.index(r[0]) if r[0] in INIT_ORDER else 99, r[1]))
    table, missing = {}, []
    for r in rows:
        for fr in fracs:
            blobs = cells.get((r[0], r[1], fr), {})
            vals = [metric_of(blobs[s], args.metric, args.split) for s in args.seeds if s in blobs]
            absent = [s for s in args.seeds if s not in blobs]
            if absent and blobs:
                missing.append((r, fr, absent))
            table[(r, fr)] = agg(vals)
    best_per_col = {}
    for fr in fracs:
        cands = [(table[(r, fr)][0], r) for r in rows if table[(r, fr)] is not None]
        best_per_col[fr] = max(cands)[1] if cands else None

    label = {"auroc": "AUROC", "auprc": "AUPRC", "onset_auroc": "AUROC (onset)",
             "onset_auprc": "AUPRC (onset)"}.get(args.metric, args.metric)
    print(f"\n{label} on {args.split}, mean ± std over seeds {args.seeds} "
          f"(* = best per column)\n")
    head = ["init / regime"] + [f"{int(round(fr*100))}% labels" for fr in fracs]
    if args.latex:
        print("\\begin{tabular}{l" + "c" * len(fracs) + "}\n\\toprule")
        print(" & ".join(head) + " \\\\\n\\midrule")
        for r in rows:
            name = f"{INIT_LABEL.get(r[0], r[0])}" + (" + JEPA aux" if r[1] > 0 else "")
            cellstrs = [fmt(table[(r, fr)], args.precision, latex=True,
                            best=(best_per_col[fr] == r)) for fr in fracs]
            print(" & ".join([name] + cellstrs) + " \\\\")
        print("\\bottomrule\n\\end{tabular}")
    else:
        w = max(len(h) for h in head) + 2
        print("".join(h.ljust(w + 8) if i else h.ljust(28) for i, h in enumerate(head)))
        for r in rows:
            name = f"{INIT_LABEL.get(r[0], r[0])}" + (" +aux" if r[1] > 0 else "")
            cellstrs = [fmt(table[(r, fr)], args.precision,
                            best=(best_per_col[fr] == r)).ljust(w + 8) for fr in fracs]
            print(name.ljust(28) + "".join(cellstrs))
    for r, fr, absent in missing:
        print(f"[warn] {r[0]} aux={r[1]} frac={fr}: missing seeds {absent}")

    # Label efficiency: smallest fraction at which each row reaches the
    # random-init full-label mean. The headline number if pretraining helps.
    ref_rows = [r for r in rows if r[0] == "random"]
    if ref_rows and 1.0 in fracs and table.get((ref_rows[0], 1.0)):
        ref = table[(ref_rows[0], 1.0)][0]
        print(f"\nfraction needed to match random-init @100% ({ref:.{args.precision}f}):")
        for r in rows:
            hit = next((fr for fr in fracs
                        if table[(r, fr)] is not None and table[(r, fr)][0] >= ref), None)
            name = f"{INIT_LABEL.get(r[0], r[0])}" + (" +aux" if r[1] > 0 else "")
            print(f"  {name:<28} {'never' if hit is None else f'{int(round(hit*100))}%'}")

    if args.json_out:
        out = {f"{r[0]}|aux{r[1]}|f{fr}": table[(r, fr)] for r in rows for fr in fracs}
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n[written] {args.json_out}")


if __name__ == "__main__":
    main()
