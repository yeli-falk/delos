#!/usr/bin/env bash
# Fetch the CHASE dynamic-label benchmarks (Kim et al., NeurIPS'26; Hugging
# Face dataset jhkim611/CHASE-datasets, CC BY-NC 4.0) into dataset/chase/ for
# src/custom_codes/chase_dataset.py:
#     DBLP-S.pkl (3.0 GB)  GossipCop.pkl (0.8 GB)  Music.pkl (0.1 GB)
#   bash scripts/dblp/fetch_chase.sh            # all three
#   bash scripts/dblp/fetch_chase.sh Music      # one file (stem of the .pkl)
# Run from a LOGIN node (needs the network). Plain HTTPS through wget / curl
# (resumable: rerun after an interruption); no python package needed. Falls
# back to huggingface_hub only when neither tool exists. Verifies SHA256SUMS.
set -euo pipefail
cd "$(dirname "$0")/../.."
DIR=dataset/chase; mkdir -p "$DIR"
REPO=jhkim611/CHASE-datasets
BASE="https://huggingface.co/datasets/$REPO/resolve/main"
if [[ $# -ge 1 ]]; then
    FILES=(); for F in "$@"; do FILES+=("${F%.pkl}.pkl"); done
else
    FILES=(DBLP-S.pkl GossipCop.pkl Music.pkl)
fi
FILES+=(SHA256SUMS README.md)

download() {   # url dest
    if command -v wget >/dev/null 2>&1; then
        wget -c -q --show-progress -O "$2" "$1"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -C - --fail --progress-bar -o "$2" "$1"
    else
        return 2
    fi
}

for F in "${FILES[@]}"; do
    if [[ "$F" == *.pkl && -s "$DIR/$F" && ! -s "$DIR/$F.part" ]]; then
        echo "[chase] have $DIR/$F"; continue
    fi
    echo "[chase] $BASE/$F"
    if download "$BASE/$F" "$DIR/$F.part"; then
        mv "$DIR/$F.part" "$DIR/$F"
    else
        # No wget / curl: use huggingface_hub from whichever python has it.
        PY=""
        for CAND in "${PY_OVERRIDE:-}" "$HOME/.conda/envs/cjepa/bin/python" "$(command -v python3 || true)" "$(command -v python || true)"; do
            [[ -n "$CAND" && -x "$CAND" ]] && "$CAND" -c "import huggingface_hub" 2>/dev/null && { PY=$CAND; break; }
        done
        [[ -n "$PY" ]] || { echo "[chase] need wget or curl, or a python with huggingface_hub (pip install huggingface_hub)" >&2; exit 5; }
        "$PY" - "$REPO" "$DIR" "$F" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo, out, f = sys.argv[1:]
hf_hub_download(repo, f, repo_type="dataset", local_dir=out)
PY
    fi
done

# Verify the pickles we asked for against the release checksums.
if [[ -s "$DIR/SHA256SUMS" ]] && command -v sha256sum >/dev/null 2>&1; then
    PAT=""
    for F in "${FILES[@]}"; do [[ "$F" == *.pkl ]] && PAT="${PAT:+$PAT|}${F//./\\.}"; done
    (cd "$DIR" && grep -E "$PAT" SHA256SUMS | sha256sum -c -) || {
        echo "[chase] checksum mismatch -- delete the bad file and rerun" >&2; exit 4; }
fi
ls -la "$DIR"
