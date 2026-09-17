# shellcheck shell=bash
# =============================================================================
# SINGLE SOURCE OF TRUTH for dataset + shared-run configuration.
#
# Every launcher sources this instead of hard-coding BINS / EDGE_FEATS / etc.
# If two scripts disagree on the node subset, the binning, the feature set, the
# window geometry or the encoder, their result tables silently describe
# DIFFERENT experiments and cannot be merged. That happened before: the MOOC
# showcase ran at 3000x97 users x items with edge features on a `tgcn` encoder
# while the baseline comparison ran at 2000x500 with edge features off on
# `gcn`, so the two tables were never comparable.
#
# Usage — set DATASET (and TASK) BEFORE sourcing:
#
#     DATASET=jodie_mooc
#     TASK=state_probe                     # or link_pred
#     source "$REPO_ROOT/scripts/dblp/_dataset_config.sh"
#     ... $DATA_ARG ...  $WINDOW_ARG ...  $ENCODER_ARG ...  $EARLY_STOP_ARG ...
#
# Every value stays env-overridable (`BINS=60 sbatch ...`), so ablations are
# still one variable away — but the DEFAULT is identical everywhere.
# =============================================================================

TASK=${TASK:-state_probe}

# --- shared across every dataset and every method ----------------------------
# The encoder is `gcn` for ALL methods. CLDG / DDGCL / IDOL each encode ONE
# aggregated snapshot (a T=1 sequence), so a recurrent encoder such as `tgcn`
# would get no temporal signal at pretrain time yet be handed T=history_size at
# probe time — a train/probe input-shape shift that penalises the baselines for
# a reason unrelated to their objective. `gcn` is shape-agnostic, so every
# method trains and probes on the same footing. (DySubC encodes k-node
# subgraphs as T=1 sequences through the same `gcn`, so it is covered too.)
#
# One exception, by design: the MaskDGNN arm passes `--encoder_kind maskdgnn`
# AFTER $ENCODER_ARG (argparse keeps the last occurrence). That encoder is this
# same GCN (same hidden / out dims) plus MaskDGNN's adaptive frequency layer
# along the window -- the layer is the method's representation learner, so it
# has to be part of what the probe sees. Everything else stays shared.
# The DyG2Vec arm does the same with `--encoder_kind dyg2vec` (the paper's
# window temporal attention encoder is the method).
ENC_KIND=${ENC_KIND:-gcn}
ENC_HIDDEN=${ENC_HIDDEN:-128}
NODE_DIM=${NODE_DIM:-128}

# Window geometry. Every method must see the same history length and forecast
# horizon or the "all" / per-horizon rows do not line up.
#
# Applying the fallback here would defeat the per-dataset defaults set further
# down (TASK=regression), because `${HIST:-12}` can no longer tell "caller did
# not set it" from "the generic default already filled it in". So remember what
# the caller actually passed, and only fall back at the end.
_HIST_FROM_CALLER=${HIST:-}
_NPRED_FROM_CALLER=${NPRED:-}

# Seeds. SEED is the single-run default; SEEDS is the multi-seed sweep list.
# Same list everywhere so per-method means are over the same seeds.
SEED=${SEED:-42}
SEEDS=${SEEDS:-"42 43 44 45 46"}

# Model selection. Every trainer (DyG-JEPA, CLDG, DDGCL, IDOL, JODIE) restores
# its best-val-loss weights and stops after PATIENCE non-improving epochs.
PATIENCE=${PATIENCE:-20}
MIN_DELTA=${MIN_DELTA:-0.0}

# --- per-dataset ------------------------------------------------------------
# JODIE datasets (wikipedia / reddit / mooc) are ALWAYS used at their full
# size; there is no node-subset knob. Sizes for reference:
#     mooc       7047 users x   97 items
#     wikipedia  8227 users x 1000 items
#     reddit    10000 users x  984 items
# The snapshot adjacency is dense (T, N, N) so memory grows ~N^2 — lower
# BATCH on the bigger two if you OOM.
# TASK=state_probe  (node classification; the JODIE/MOOC family)
#   Edge features on: the state label is a per-user event, so coverage of
#   users is what the probe needs.
# TASK=link_pred    (MRR/Hits@k)
#   Edge features off (historical setting); the probe scores 99 negatives per
#   positive edge per window, so it is the slower of the two at full N.
case "$TASK" in
    state_probe)
        BINS=${BINS:-30}
        EDGE_FEATS=${EDGE_FEATS:-1}
        ;;
    link_pred)
        BINS=${BINS:-30}
        TOPK=${TOPK:-1000}                       # reddit-hyperlinks subreddits
        REDDIT_BINS=${REDDIT_BINS:-36}           # ~monthly over the SNAP span
        EDGE_FEATS=${EDGE_FEATS:-0}
        ;;
    regression)
        # Fixed-node-set signal forecasting (england_covid, metr_la): no
        # binning knobs — the whole graph is used.
        BINS=${BINS:-0}
        EDGE_FEATS=${EDGE_FEATS:-0}
        # Window geometry follows each dataset's PUBLISHED convention, so our
        # numbers sit on the same axis as the literature's:
        #   metr_la        12 history -> 12 horizon at native 5-min resolution,
        #                  scored at +3/+6/+12 (15/30/60 min). This is the DCRNN
        #                  (Li et al. 2018) protocol that STGCN, Graph WaveNet
        #                  and ASTGCN all report against.
        #   england_covid  8 daily lags, the PyG-Temporal / Panagopoulos et al.
        #                  (AAAI'21) EnglandCovid setting.
        # Every method in a table sources this, so none of them can quietly run
        # a different history length or horizon than the others.
        case "$DATASET" in
            metr_la)
                HIST=${_HIST_FROM_CALLER:-12}; NPRED=${_NPRED_FROM_CALLER:-12}
                HORIZONS=${HORIZONS:-"2 5 11"}       # 0-indexed +3/+6/+12 steps
                REG_BATCH=${REG_BATCH:-64}
                ;;
            england_covid)
                HIST=${_HIST_FROM_CALLER:-8}; NPRED=${_NPRED_FROM_CALLER:-3}
                HORIZONS=${HORIZONS:-"0 1 2"}
                REG_BATCH=${REG_BATCH:-8}
                ;;
            *)
                HIST=${_HIST_FROM_CALLER:-8}; NPRED=${_NPRED_FROM_CALLER:-3}
                HORIZONS=${HORIZONS:-"0"}
                REG_BATCH=${REG_BATCH:-8}
                ;;
        esac
        ;;
    *)
        echo "[error] TASK must be 'state_probe', 'link_pred' or 'regression', " \
             "got '$TASK'" >&2
        exit 2 ;;
esac

# --- build DATA_ARG ---------------------------------------------------------
# Normalise the bare JODIE names (`mooc`) to the --data_source spelling.
case "$DATASET" in
    mooc|reddit|wikipedia) DATASET="jodie_${DATASET}" ;;
esac

case "$DATASET" in
    dblp)
        if [[ ! -f dataset/dblp/edges.csv ]]; then
            echo "[error] dataset/dblp/edges.csv not found" >&2
            exit 3
        fi
        DATA_ARG="--data_source dblp --data_path dataset/dblp/edges.csv"
        ;;
    jodie_mooc|jodie_reddit|jodie_wikipedia)
        VARIANT=${DATASET#jodie_}
        JODIE_CSV="dataset/jodie/${VARIANT}.csv"
        DATA_ARG="--data_source $DATASET --jodie_data_path $JODIE_CSV \
--jodie_num_bins $BINS"
        ;;
    reddit_body|reddit_title)
        VARIANT=${DATASET#reddit_}
        REDDIT_TSV="dataset/reddit/soc-redditHyperlinks-${VARIANT}.tsv"
        DATA_ARG="--data_source $DATASET --reddit_data_path $REDDIT_TSV \
--reddit_num_bins ${REDDIT_BINS:-36} --reddit_top_k ${TOPK:-1000}"
        ;;
    england_covid|metr_la)
        DATA_ARG="--data_source $DATASET"
        ;;
    synthetic)
        DATA_ARG="--synthetic --synthetic_nodes ${SYN_NODES:-500} \
--synthetic_years ${SYN_YEARS:-30}"
        ;;
    synthetic_dyn)
        # Latent-driven generator (dynamic_graph_generator.py) configured for
        # the dense node-STATE task: the community is the anchor z is closest
        # to (a readout of the latent), so the state at t+1 is knowable from
        # the trajectory and a forecast has real headroom over copy-last.
        # anchor_scale / ou_sigma are set for a ~5% per-step switch rate and
        # ~16% oracle error reduction over persistence (state_headroom, h=3);
        # the loader prints that diagnostic on the test split -- read it.
        # No edge features: the DTDG snapshots are binary.
        EDGE_FEATS=0
        DATA_ARG="--data_source synthetic_dyn \
--synth_nodes ${SYN_NODES:-200} --synth_steps ${SYN_STEPS:-300} \
--synth_label_mode ${SYN_LABEL_MODE:-nearest_anchor} \
--synth_state_target ${SYN_STATE_TARGET:-community} \
--synth_hysteresis ${SYN_HYST:-0.0} \
--synth_anchor_scale ${SYN_ANCHOR:-1.0} --synth_ou_sigma ${SYN_OU_SIGMA:-0.35} \
--synth_momentum ${SYN_MOMENTUM:-0.8} --synth_ou_theta ${SYN_THETA:-0.15} \
--synth_node_noise ${SYN_NODE_NOISE:-0.10} --synth_target_noise ${SYN_TARGET_NOISE:-0.0} \
--synth_communities ${SYN_K:-4} --synth_d_latent ${SYN_DLAT:-8} --synth_feat_dim ${SYN_FEAT_DIM:-16} \
--synth_beta ${SYN_BETA:-1.0} --synth_density ${SYN_DENSITY:-0.05} \
--synth_persistence ${SYN_PERSISTENCE:-0.6} --synth_diffusion ${SYN_DIFFUSION:-0.3}"
        ;;
    *)
        echo "[error] unsupported DATASET='$DATASET'. Supported: dblp, " \
             "jodie_{mooc,reddit,wikipedia}, reddit_{body,title}, synthetic, " \
             "synthetic_dyn" >&2
        exit 2 ;;
esac

# Edge features change the encoder's INPUT DIM, so this flag must be identical
# for every method in a table — never set it for one method only.
[[ "$EDGE_FEATS" == "1" ]] && DATA_ARG="$DATA_ARG --with_edge_features"

# Generic fallback for the tasks that set no per-dataset window (state_probe,
# link_pred). The regression branch has already pinned its own values above.
HIST=${HIST:-3}
NPRED=${NPRED:-1}

WINDOW_ARG="--history_size $HIST --num_preds $NPRED"
ENCODER_ARG="--encoder_kind $ENC_KIND --encoder_hidden $ENC_HIDDEN --node_dim $NODE_DIM"
EARLY_STOP_ARG="--early_stop_patience $PATIENCE --early_stop_min_delta $MIN_DELTA"

echo "[config] TASK=$TASK DATASET=$DATASET ENC_KIND=$ENC_KIND BINS=$BINS " \
     "EDGE_FEATS=$EDGE_FEATS HIST=$HIST NPRED=$NPRED " \
     "PATIENCE=$PATIENCE SEED=$SEED"
echo "[config] DATA_ARG=$DATA_ARG"
