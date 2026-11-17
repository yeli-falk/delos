#!/usr/bin/env bash
# Fetch NYC TLC monthly trip records into dataset/nyc_taxi/ for
# src/custom_codes/nyc_taxi_dataset.py.
#   bash scripts/dblp/fetch_nyc_taxi.sh 2019-01 2019-02 2019-03
#   KIND=green bash scripts/dblp/fetch_nyc_taxi.sh 2019-01
# Run from a LOGIN node. Yellow 2019: ~100 MB and ~7M trips per month (the
# loader aggregates one month at a time and caches an .npz). Reading parquet
# needs pyarrow in the training env: pip install pyarrow
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ $# -ge 1 ]] || { echo "usage: fetch_nyc_taxi.sh <YYYY-MM> [<YYYY-MM> ...]" >&2; exit 2; }
KIND="${KIND:-yellow}"
DIR=dataset/nyc_taxi; mkdir -p "$DIR"
download() {
    if command -v wget >/dev/null 2>&1 && wget -q -O "$2" "$1"; then return 0; fi
    if command -v curl >/dev/null 2>&1 && curl -sSfL -o "$2" "$1"; then return 0; fi
    python - "$1" "$2" <<'PY'
import sys, urllib.request
with urllib.request.urlopen(sys.argv[1]) as r, open(sys.argv[2], "wb") as f: f.write(r.read())
PY
}
for M in "$@"; do
    F="$DIR/${KIND}_tripdata_${M}.parquet"
    if [[ -s "$F" ]]; then echo "[nyc] have $F"; continue; fi
    URL="https://d37ci6vzurychx.cloudfront.net/trip-data/${KIND}_tripdata_${M}.parquet"
    echo "[nyc] $URL"
    download "$URL" "$F.part" && [[ -s "$F.part" ]] && mv "$F.part" "$F" \
        || { rm -f "$F.part"; echo "[warn] failed: $URL" >&2; }
done
ls -la "$DIR"
