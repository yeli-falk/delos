#!/usr/bin/env bash
# Unpack the CLDG-release graphs (DBLP-10 / Patent-6) from the Google-Drive
# zips into the layout src/custom_codes/cldg_format_dataset.py reads:
#
#     dataset/cldg/<name>/<name>.txt        src dst t   (t normalised to [0,1])
#     dataset/cldg/<name>/node2label.txt    node label
#     dataset/cldg/<name>/feature.txt       128 floats per node (sinusoidal
#                                           node-index code, CLDG's input)
#
# Only those three files are extracted. label.txt duplicates node2label.txt
# (sorted by node) and "MVC Features/" holds another method's precomputed
# embeddings; both stay in the zip. Idempotent: re-running only refreshes
# missing files. The loaders cache .npz / .npy next to the text files on
# first use.
#
# Usage (repo root, where the zips were dropped in dataset/):
#     bash scripts/dblp/extract_cldg_datasets.sh [dataset/dblp-*.zip dataset/patent-*.zip]
set -euo pipefail
cd "$(dirname "$0")/../.."

ZIPS=("$@")
if (( ${#ZIPS[@]} == 0 )); then
    ZIPS=(dataset/dblp-*.zip dataset/patent-*.zip)
fi
for z in "${ZIPS[@]}"; do
    [[ -f "$z" ]] || { echo "[warn] $z not found, skipping" >&2; continue; }
    [[ -s "$z" ]] || { echo "[error] $z is empty (download unfinished?)" >&2; exit 1; }
    folder=$(basename "$z"); folder=${folder%%-*}    # dblp-2026...zip -> dblp
    name=$(echo "$folder" | tr 'A-Z' 'a-z')           # arXivAI -> arxivai
    case "$name" in dblp|patent) ;; *) echo "[error] unknown archive $z" >&2; exit 1 ;; esac
    # the edge file inside the zip may differ in case (arXivAI/arxivAI.txt)
    edge=$(unzip -Z1 "$z" | grep -i "^$folder/$folder\.txt$" | head -1)
    [[ -n "$edge" ]] || { echo "[error] no $folder/$folder.txt in $z" >&2; exit 1; }
    TMP=$(mktemp -d)
    unzip -q -o "$z" "$edge" "$folder/node2label.txt" "$folder/feature.txt" -d "$TMP"
    mkdir -p "dataset/cldg/$name"
    [[ -f "dataset/cldg/$name/$name.txt" ]] || mv -f "$TMP/$edge" "dataset/cldg/$name/$name.txt"
    for f in node2label.txt feature.txt; do
        [[ -f "dataset/cldg/$name/$f" ]] || mv -f "$TMP/$folder/$f" "dataset/cldg/$name/$f"
    done
    rm -rf "$TMP"
    printf '[cldg] %-8s %s edges  %s labelled nodes  ->  dataset/cldg/%s/\n' \
        "$name" "$(wc -l < "dataset/cldg/$name/$name.txt")" \
        "$(wc -l < "dataset/cldg/$name/node2label.txt")" "$name"
done
