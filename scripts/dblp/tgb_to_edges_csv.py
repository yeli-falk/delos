"""Convert a TGB edgelist CSV to our `(src, dst, year)` format.

TGB files look like:
    u,i,ts,label,idx
    0,1,1356998400,1,0
    ...

`ts` is Unix seconds for most TGB link datasets. We discretise to years
(using the calendar year of the timestamp) so the snapshots feed into our
existing `DBLPDynamicGraphDataset(data_path=...)` loader unchanged.

Usage:
    python scripts/dblp/tgb_to_edges_csv.py \
        --input /workdir/.../tgbl-review_edgelist_v2.csv \
        --output dataset/dblp/edges.csv \
        --time_format unix \
        --min_year 2005 --max_year 2018
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Path to the TGB edgelist CSV.")
    p.add_argument("--output", required=True,
                   help="Destination CSV (`src,dst,year` rows, no header).")
    p.add_argument("--time_format", choices=["unix", "raw"], default="unix",
                   help="`unix` -> seconds since epoch (default for TGB); "
                        "`raw` -> use the timestamp column directly as the year "
                        "bucket (already-integer year datasets).")
    p.add_argument("--columns", default="u,i,ts",
                   help="Comma-separated column names: src,dst,timestamp.")
    p.add_argument("--min_year", type=int, default=None)
    p.add_argument("--max_year", type=int, default=None)
    p.add_argument("--drop_self_loops", action="store_true", default=True)
    p.add_argument("--make_undirected", action="store_true",
                   help="If set, also emit the reverse edge.")
    args = p.parse_args()

    src_col, dst_col, ts_col = args.columns.split(",")

    print(f"[load] reading {args.input}")
    df = pd.read_csv(args.input)
    print(f"       columns: {list(df.columns)}")
    print(f"       rows:    {len(df):,}")

    for c in (src_col, dst_col, ts_col):
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not in CSV (available: {list(df.columns)})")

    if args.time_format == "unix":
        years = pd.to_datetime(df[ts_col], unit="s", errors="coerce").dt.year
    else:
        years = df[ts_col].astype(int)

    keep = years.notna()
    if args.min_year is not None:
        keep &= years >= args.min_year
    if args.max_year is not None:
        keep &= years <= args.max_year
    df = df.loc[keep].copy()
    years = years.loc[keep].astype(int)
    print(f"[filter] {len(df):,} rows after year filter "
          f"({args.min_year or '-inf'} .. {args.max_year or '+inf'})")

    if args.drop_self_loops:
        not_loop = df[src_col] != df[dst_col]
        df = df.loc[not_loop]
        years = years.loc[not_loop]
        print(f"[filter] {len(df):,} rows after dropping self-loops")

    out = pd.DataFrame({
        "src": df[src_col].astype(int).values,
        "dst": df[dst_col].astype(int).values,
        "year": years.values,
    })
    if args.make_undirected:
        rev = out.rename(columns={"src": "dst", "dst": "src"})[["src", "dst", "year"]]
        out = pd.concat([out, rev], ignore_index=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, header=False)
    print(f"[done] wrote {len(out):,} rows to {out_path}")
    print(f"       year range: {out['year'].min()} .. {out['year'].max()}")
    print(f"       distinct nodes: {len(set(out['src']) | set(out['dst'])):,}")
    print(f"       distinct years: {out['year'].nunique()}")


if __name__ == "__main__":
    main()
