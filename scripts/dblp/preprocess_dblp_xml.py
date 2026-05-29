"""Stream-parse the official DBLP XML dump into our `(src, dst, year)` CSV.

The dump (https://dblp.org/xml/dblp.xml.gz) is ~3 GB gzipped / ~30 GB
uncompressed. We use `lxml.etree.iterparse` to stream it in constant memory.
For each `<article>`, `<inproceedings>`, etc. we extract the author list and
year, then emit a coauthorship edge for every author pair.

The output is exactly the format `DBLPDynamicGraphDataset(data_path=...)`
already consumes:
    src,dst,year       (no header)

Usage:
    pip install lxml
    wget https://dblp.org/xml/dblp.xml.gz -O dataset/dblp/dblp.xml.gz
    python scripts/dblp/preprocess_dblp_xml.py \
        --xml_path dataset/dblp/dblp.xml.gz \
        --out_csv dataset/dblp/edges.csv \
        --min_year 1990 --max_year 2023 \
        --min_author_papers 5
"""
from __future__ import annotations

import argparse
import gzip
import time
from itertools import combinations
from pathlib import Path


PUBLICATION_TAGS = ("article", "inproceedings", "incollection", "book",
                    "proceedings")


def stream_publications(xml_path: str, include_tags: tuple, min_year: int,
                        max_year: int, progress_every: int = 200_000):
    """Yield (authors: list[str], year: int) per publication."""
    from lxml import etree as ET

    is_gz = str(xml_path).endswith(".gz")
    opener = gzip.open if is_gz else open
    t0 = time.time()
    count_total = 0
    count_yielded = 0
    with opener(xml_path, "rb") as f:
        # recover=True swallows DBLP's DTD-defined HTML entities (e.g. &auml;)
        # without aborting. We don't resolve them, so author names containing
        # special characters survive as literal entity strings -- functional
        # for graph construction; cosmetically wrong if you ever print names.
        context = ET.iterparse(
            f, events=("end",), tag=include_tags,
            recover=True, load_dtd=False, resolve_entities=False, huge_tree=True,
        )
        for _event, elem in context:
            count_total += 1
            authors = [a.text for a in elem.findall("author") if a.text]
            year_elem = elem.find("year")
            year = None
            if year_elem is not None and year_elem.text:
                txt = year_elem.text.strip()
                if txt.isdigit():
                    year = int(txt)
            if year is not None and min_year <= year <= max_year and len(authors) >= 2:
                yield authors, year
                count_yielded += 1
            # Free memory: clear element + all preceding siblings still attached
            # to the parent.
            elem.clear()
            parent = elem.getparent()
            if parent is not None:
                while parent.index(elem) > 0:
                    del parent[0]
            if count_total % progress_every == 0:
                rate = count_total / max(time.time() - t0, 1e-6)
                print(f"  ... read {count_total:,} pubs "
                      f"({count_yielded:,} kept) -- {rate:,.0f} pubs/s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml_path", required=True,
                   help="dblp.xml or dblp.xml.gz")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--min_year", type=int, default=1990)
    p.add_argument("--max_year", type=int, default=2023)
    p.add_argument("--min_author_papers", type=int, default=5,
                   help="Drop authors with fewer than this many qualifying papers.")
    p.add_argument("--include_tags", default="article,inproceedings,incollection",
                   help="Comma-separated DBLP entry tags to include.")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many publications (for testing).")
    p.add_argument("--names_csv", default=None,
                   help="Optional path: dump `id,author_name` mapping here.")
    args = p.parse_args()

    include_tags = tuple(t.strip() for t in args.include_tags.split(",") if t.strip())
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[stream] reading {args.xml_path}")
    print(f"         tags={include_tags}  years={args.min_year}..{args.max_year}")

    # First pass: collect publications into compact form (author IDs + year).
    # IDs are assigned by first appearance so we can densify later.
    name_to_id: dict[str, int] = {}
    publications: list[tuple[int, tuple[int, ...]]] = []

    for authors, year in stream_publications(
        args.xml_path, include_tags, args.min_year, args.max_year,
    ):
        ids = []
        for name in authors:
            i = name_to_id.get(name)
            if i is None:
                i = len(name_to_id)
                name_to_id[name] = i
            ids.append(i)
        publications.append((year, tuple(ids)))
        if args.limit and len(publications) >= args.limit:
            break

    n_pubs = len(publications)
    n_authors_seen = len(name_to_id)
    print(f"[stream] kept {n_pubs:,} publications, {n_authors_seen:,} distinct authors")

    # Author paper count for the --min_author_papers filter.
    if args.min_author_papers > 0:
        paper_count = [0] * n_authors_seen
        for _, ids in publications:
            for i in ids:
                paper_count[i] += 1
        kept_old_ids = [i for i, c in enumerate(paper_count) if c >= args.min_author_papers]
        print(f"[filter] keeping {len(kept_old_ids):,}/{n_authors_seen:,} authors "
              f"with >= {args.min_author_papers} papers")
    else:
        kept_old_ids = list(range(n_authors_seen))

    # Densify IDs (so the final graph has contiguous 0..N-1).
    old_to_new = {old: new for new, old in enumerate(kept_old_ids)}
    print(f"[filter] final node count: {len(old_to_new):,}")

    # Pass 2: emit edges (no per-year deduplication; the dataset loader's
    # symmetric-bool adjacency dedups automatically when materialised).
    n_edges = 0
    n_pubs_kept = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        for year, ids in publications:
            new_ids = [old_to_new[i] for i in ids if i in old_to_new]
            if len(new_ids) < 2:
                continue
            for i, j in combinations(new_ids, 2):
                a, b = (i, j) if i < j else (j, i)
                f.write(f"{a},{b},{year}\n")
                n_edges += 1
            n_pubs_kept += 1

    print(f"[done] wrote {n_edges:,} edges to {out_path} in {time.time()-t0:.1f}s")
    print(f"       {n_pubs_kept:,} publications contributed at least one edge")

    if args.names_csv:
        new_to_name = {new: name for name, old in name_to_id.items()
                       for new in [old_to_new.get(old)] if new is not None}
        with open(args.names_csv, "w") as f:
            f.write("id,name\n")
            for nid in sorted(new_to_name):
                # Names may contain commas — quote.
                nm = new_to_name[nid].replace('"', '""')
                f.write(f'{nid},"{nm}"\n')
        print(f"[done] author-name table -> {args.names_csv}")


if __name__ == "__main__":
    main()
