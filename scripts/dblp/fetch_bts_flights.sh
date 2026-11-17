#!/usr/bin/env bash
# Fetch BTS "Reporting Carrier On-Time Performance" monthly zips into
# dataset/bts/ for src/custom_codes/bts_flights_dataset.py.
#   bash scripts/dblp/fetch_bts_flights.sh 2019            # one year
#   bash scripts/dblp/fetch_bts_flights.sh 2018 2019 2022  # several
# Run from a LOGIN node. ~25 MB per monthly zip (~200 MB csv unpacked; the
# loader reads the zip directly and caches an .npz). transtats occasionally
# refuses non-browser clients, hence the user agent.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ $# -ge 1 ]] || { echo "usage: fetch_bts_flights.sh <year> [<year> ...]" >&2; exit 2; }
DIR=dataset/bts; mkdir -p "$DIR"
UA="Mozilla/5.0 (X11; Linux x86_64)"
download() {
    if command -v wget >/dev/null 2>&1 && wget -q --user-agent="$UA" -O "$2" "$1"; then return 0; fi
    if command -v curl >/dev/null 2>&1 && curl -sSL -A "$UA" -o "$2" "$1"; then return 0; fi
    python - "$1" "$2" "$UA" <<'PY'
import sys, urllib.request
req = urllib.request.Request(sys.argv[1], headers={"User-Agent": sys.argv[3]})
with urllib.request.urlopen(req) as r, open(sys.argv[2], "wb") as f: f.write(r.read())
PY
}
for Y in "$@"; do
    for M in $(seq 1 12); do
        F="$DIR/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_${Y}_${M}.zip"
        if [[ -s "$F" ]]; then echo "[bts] have $F"; continue; fi
        URL="https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_${Y}_${M}.zip"
        echo "[bts] $URL"
        download "$URL" "$F.part" && [[ -s "$F.part" ]] && mv "$F.part" "$F" \
            || { rm -f "$F.part"; echo "[warn] failed: $URL (month missing or not yet published)" >&2; }
    done
done
ls -la "$DIR"
