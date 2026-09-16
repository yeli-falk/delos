#!/usr/bin/env bash
# Fetch a preprocessed temporal-graph dataset and convert to edges.csv.
#
# Three sources, pick whichever is reachable from your environment.
# After downloading, run pickled_to_csv.py on the resulting graphs.pkl.

set -e
DEST=dataset/dblp
mkdir -p "$DEST"

cat <<'EOF'
=============================================================================
Preprocessed temporal-graph dataset sources (manual download required)
=============================================================================

(A) DySAT (Sankar et al., WSDM 2020) -- closest to "DBLP-like"
    Repo:    https://github.com/aravindsankar28/DySAT
    Data:    repo's data/ directory (Enron, UCI, Yelp, ML-10M variants)
    Format:  pickled list of networkx.Graph snapshots
    Size:    a few hundred MB

    git clone https://github.com/aravindsankar28/DySAT.git /tmp/DySAT
    cp /tmp/DySAT/data/Enron_16/graphs.pkl dataset/dblp/graphs.pkl

(B) EvolveGCN (Pareja et al., AAAI 2020)
    Repo:    https://github.com/IBM/EvolveGCN
    Data:    referenced via data_loader.py; some files in repo, others linked
    Format:  npz / pickle, see their data_loader for parsing
    Size:    varies (Bitcoin-OTC small, DBLP medium)

(C) TGB (Huang et al., NeurIPS 2023) -- maintained, auto-download
    Install: pip install py-tgb
    Usage:
        python -c "from tgb.linkproppred.dataset import LinkPropPredDataset; \
                   d = LinkPropPredDataset(name='tgbl-review', root='dataset/dblp/tgb')"
    Output: Numpy arrays of (src, dst, ts, label) -- you'd write a tiny
    converter mapping their integer timestamps to "year buckets".

=============================================================================

After fetching one of the above into dataset/dblp/graphs.pkl, run:

    python scripts/dblp/pickled_to_csv.py \\
        --input dataset/dblp/graphs.pkl \\
        --output dataset/dblp/edges.csv \\
        --year_start 2000 \\
        --undirected

Then the DBLP training script will pick it up:

    python src/train/train_dyg_jepa.py --no_attention_mask \\
        --data_path dataset/dblp/edges.csv \\
        --device cuda --epochs 5
EOF
