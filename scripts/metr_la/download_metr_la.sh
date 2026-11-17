#!/usr/bin/env bash
# Fetch METR-LA from the Hugging Face mirror (witgaw/METR-LA).
#
# This is a pre-windowed parquet release (history = 12, future = 12,
# 5-min steps), so it's *not* a drop-in for the original DCRNN .h5 layout
# but is much easier to download. The MetrLADataset loader detects this
# layout and pivots back into (T, N, F) at load time.

set -e
DEST=dataset/metr_la
mkdir -p "$DEST"

PY=${PYTHON:-python}
if [[ -z "${PYTHON:-}" && -x "$HOME/.conda/envs/cjepa/bin/python" ]]; then
    PY="$HOME/.conda/envs/cjepa/bin/python"
fi

if ! "$PY" -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "Installing huggingface_hub..."
    "$PY" -m pip install --quiet huggingface_hub
fi

"$PY" - <<EOF
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="witgaw/METR-LA",
    repo_type="dataset",
    local_dir="$DEST",
    local_dir_use_symlinks=False,
    allow_patterns=["*.parquet", "sensor_graph/*"],
)
print("Downloaded to:", path)
EOF

echo "Files in $DEST:"
ls -la "$DEST"
