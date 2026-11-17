#!/usr/bin/env bash
# Fetch one TGB node-property dataset's raw CSVs (Huang et al., NeurIPS'23)
# into dataset/tgb/tgbn-<name>/ for src/custom_codes/tgbn_dataset.py.
#   bash scripts/dblp/fetch_tgbn.sh trade|genre|reddit|token
# Run from a LOGIN node (compute nodes usually have no internet). Sizes:
# trade ~12 MB zip; genre ~0.3 GB; reddit ~0.6 GB; token ~1.5 GB (unpacked
# CSVs are 3-5x larger). The loader caches an .npz next to the CSVs on first
# use. URLs are py-tgb 2.3.0's alliancecan mirror (tgb/utils/info.py).
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME="${1:?usage: fetch_tgbn.sh trade|genre|reddit|token}"
case "$NAME" in trade|genre|reddit|token) ;; *) echo "[error] unknown dataset '$NAME'" >&2; exit 2 ;; esac
URL="https://object-arbutus.alliancecan.ca/swift/v1/14c95234f6cd4a21a47deafe20cce2a7/tgb/tgbn-${NAME}.zip"
DIR="dataset/tgb/tgbn-${NAME}"
mkdir -p "$DIR"
if [[ -f "$DIR/tgbn-${NAME}_edgelist.csv" && -f "$DIR/tgbn-${NAME}_node_labels.csv" ]]; then
    echo "[tgbn-${NAME}] already present in $DIR"; exit 0
fi
TMP=$(mktemp -d)
# Download with whatever works on this node: wget, then curl, then Python.
# (On ruche the conda env's curl loads the system libcurl and dies with
# "curl: (48) An unknown option was passed in to libcurl".)
download() {
    if command -v wget >/dev/null 2>&1 && wget -q -O "$2" "$1"; then return 0; fi
    if command -v curl >/dev/null 2>&1 && curl -sSL -o "$2" "$1"; then return 0; fi
    python - "$1" "$2" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
}
download "$URL" "$TMP/tgbn-${NAME}.zip" || { echo "[error] download failed: $URL" >&2; exit 1; }
[[ -s "$TMP/tgbn-${NAME}.zip" ]] || { echo "[error] empty download: $URL" >&2; exit 1; }
unzip -q -o "$TMP/tgbn-${NAME}.zip" -d "$TMP"
# the zip holds the CSVs at its root (or under tgbn-<name>/ in older builds)
find "$TMP" -name "tgbn-${NAME}_*.csv" -exec mv -f {} "$DIR"/ \;
rm -rf "$TMP"
ls -la "$DIR"
