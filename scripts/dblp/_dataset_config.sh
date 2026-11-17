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
# window temporal attention encoder is the method), and the DyG-WM arm with
# `--encoder_kind dygwm` (this GCN + its node / neighbourhood trajectory GRUs).
# SG-JEPA keeps the plain shared GCN: its spiking layers are its training
# apparatus, not its representation.
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
# Normalise the bare JODIE names (`mooc`) to the --data_source spelling, and
# `patent` to its CLDG-format source. (`dblp` stays the unlabelled XML-derived
# link-prediction graph; the labelled CLDG release is `cldg_dblp`.)
case "$DATASET" in
    mooc|reddit|wikipedia) DATASET="jodie_${DATASET}" ;;
    patent) DATASET="cldg_patent" ;;
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
    tgbn_trade)
        # TGB node property prediction (tgbn-trade): 255 nations, 31 native
        # yearly snapshots, label = each nation's export-share vector for the
        # target year, probed with `--task ranking` (NDCG@10, TGB's metric;
        # label_persistence = last year's vector, TGB's own baseline). No
        # binning / subset knobs: the whole graph fits. TGBN_FEATS=0 drops
        # the 5 trade-weight descriptors (the bool snapshot loses weights).
        if [[ ! -f dataset/tgb/tgbn-trade/tgbn-trade_node_labels.csv ]]; then
            echo "[error] dataset/tgb/tgbn-trade/*.csv not found;" \
                 "run: bash scripts/dblp/fetch_tgbn_trade.sh (login node)" >&2
            exit 3
        fi
        EDGE_FEATS=0
        DATA_ARG="--data_source tgbn_trade --tgbn_data_dir dataset/tgb/tgbn-trade"
        [[ "${TGBN_FEATS:-1}" == "0" ]] && DATA_ARG="$DATA_ARG --no_tgbn_weight_features"
        ;;
    bts_flights)
        # BTS airport delay propagation: the flights flown each day are the
        # edges (the graph changes with schedule / season / cancellations),
        # the label is whether an airport is in a DELAYED state that day
        # (share of departures delayed >= 15 min above BTS_THR), a recurring
        # node state. Binary state probe (default PROBE_TASK). Knobs:
        #   BTS_YEARS      "2019" (space-separated, consecutive)
        #   BTS_BIN_HOURS  24 (must divide 24; sub-daily = scheduled local hour)
        #   BTS_LABEL      share_del15 | mean_delay,  BTS_THR 0.30 (use ~15 for mean_delay)
        #   BTS_MIN_DEP    5 departures for a labelled airport-bin
        #   BTS_FEATS=0    graph statistics only (drops the delay descriptors)
        for Y in ${BTS_YEARS:-2019}; do
            if ! ls dataset/bts/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_${Y}_*.zip >/dev/null 2>&1; then
                echo "[error] dataset/bts/*_${Y}_*.zip not found; run: bash scripts/dblp/fetch_bts_flights.sh ${Y} (login node)" >&2
                exit 3
            fi
        done
        EDGE_FEATS=0
        DATA_ARG="--data_source bts_flights --bts_years ${BTS_YEARS:-2019} \
--bts_bin_hours ${BTS_BIN_HOURS:-24} --bts_label ${BTS_LABEL:-share_del15} \
--bts_threshold ${BTS_THR:-0.30} --bts_min_departures ${BTS_MIN_DEP:-5}"
        [[ "${BTS_FEATS:-1}" == "0" ]] && DATA_ARG="$DATA_ARG --no_bts_ops_features"
        ;;
    nyc_taxi)
        # NYC taxi zone congestion: the trips driven each hour are the edges
        # (pickup zone -> dropoff zone), the label is whether a zone is
        # CONGESTED in the bin (mean speed of its pickups below the zone's own
        # NYC_THR quantile over the train bins): a recurring node state with
        # daily and weekly cycles. Binary state probe. Read the loader's
        # "trivial floors" line (copy-last / same bin yesterday / last week)
        # before any table. For the history to cover a period use HIST >=
        # 24 / NYC_BIN_HOURS. Knobs:
        #   NYC_MONTHS     "2019-01 2019-02 2019-03" (space-separated, consecutive)
        #   NYC_BIN_HOURS  1 (must divide 24)
        #   NYC_KIND       yellow | green
        #   NYC_LABEL      rel_speed | abs_speed,  NYC_THR 0.25 (mph, ~10, for abs_speed)
        #   NYC_MIN_TRIPS  10 valid-speed pickups for a labelled zone-bin
        #   NYC_MIN_EDGE   1 trip for an edge (raise to sparsify the snapshots)
        #   NYC_FEATS=0    graph statistics only (drops the trip descriptors)
        for M in ${NYC_MONTHS:-2019-01 2019-02 2019-03}; do
            if ! ls dataset/nyc_taxi/${NYC_KIND:-yellow}_tripdata_${M}.* >/dev/null 2>&1; then
                echo "[error] dataset/nyc_taxi/${NYC_KIND:-yellow}_tripdata_${M}.parquet not found; run: bash scripts/dblp/fetch_nyc_taxi.sh ${M} (login node)" >&2
                exit 3
            fi
        done
        EDGE_FEATS=0
        DATA_ARG="--data_source nyc_taxi --nyc_months ${NYC_MONTHS:-2019-01 2019-02 2019-03} \
--nyc_bin_hours ${NYC_BIN_HOURS:-1} --nyc_kind ${NYC_KIND:-yellow} --nyc_label ${NYC_LABEL:-rel_speed} \
--nyc_threshold ${NYC_THR:-0.25} --nyc_min_trips ${NYC_MIN_TRIPS:-10} \
--nyc_min_edge_trips ${NYC_MIN_EDGE:-1}"
        [[ "${NYC_FEATS:-1}" == "0" ]] && DATA_ARG="$DATA_ARG --no_nyc_ops_features"
        ;;
    tgbn_genre|tgbn_reddit|tgbn_token)
        # TGB weekly interaction streams (user -> genre / subreddit / token):
        # label = the user's next-week class distribution, probed with
        # `--task ranking`. The probe reports BOTH TGB floors: label_persistence
        # (last week) and label_moving_average (mean over the history window);
        # the moving average beats every learned model on TGB's leaderboard, so
        # it is the number to beat. Knobs:
        #   TGBN_BIN_DAYS  snapshot length in days (default 7 = the label grid)
        #   TGBN_MAX_BINS  keep only the last k labelled weeks + the nodes seen
        #                  in them (time cut; needed for reddit's ~730 weeks x
        #                  11.8k nodes in the dense pipeline, e.g. 104). Token
        #                  (61.8k nodes) does not fit the dense pipeline.
        #   TGBN_FEATS=0   drop the 5 interaction-weight descriptors.
        TGBN_NAME=${DATASET#tgbn_}
        if [[ ! -f dataset/tgb/tgbn-${TGBN_NAME}/tgbn-${TGBN_NAME}_node_labels.csv ]]; then
            echo "[error] dataset/tgb/tgbn-${TGBN_NAME}/*.csv not found;" \
                 "run: bash scripts/dblp/fetch_tgbn.sh ${TGBN_NAME} (login node)" >&2
            exit 3
        fi
        EDGE_FEATS=0
        DATA_ARG="--data_source $DATASET --tgbn_data_dir dataset/tgb/tgbn-${TGBN_NAME} \
--tgbn_bin_days ${TGBN_BIN_DAYS:-7}"
        [[ -n "${TGBN_MAX_BINS:-}" ]] && DATA_ARG="$DATA_ARG --tgbn_max_bins $TGBN_MAX_BINS"
        [[ "${TGBN_FEATS:-1}" == "0" ]] && DATA_ARG="$DATA_ARG --no_tgbn_weight_features"
        ;;
    chase_dblp_s|chase_gossipcop|chase_music)
        # CHASE benchmarks (Kim et al., NeurIPS'26): snapshot graphs whose
        # node labels CHANGE over time (research area per year / >50% fake
        # news shared per week / dominant genre per month) with per-snapshot
        # node features from the release. Splits are the paper's snapshot
        # counts (9/2/3, 26/5/6, 15/4/5), set by the loader. dblp_s and
        # gossipcop are binary (default PROBE_TASK); music is a 5-class
        # dynamic state (the launchers switch to --task multiclass). Read the
        # loader's label audit (persistence floor: music 0.81, gossipcop
        # 0.96) before any table. Knobs:
        #   CHASE_TOPK   0 = full graph for all three (34,717 / 6,091 / 878
        #                nodes). Set it (e.g. 8000) to keep only the K most
        #                active nodes by summed degree; the dense (T, N, N)
        #                snapshots make full DBLP-S heavy (1.2 G cells per
        #                snapshot), so lower BATCH first if it OOMs.
        #   CHASE_FEATS  1 = append the release's node vectors (768-d BERT /
        #                779-d listening descriptors); 0 = graph statistics
        #                only (topology ablation)
        #   CHASE_NORM   1 = z-score them on the train snapshots
        #   CHASE_PCA    0 = keep every dim, else PCA to this many
        # Files: bash scripts/dblp/fetch_chase.sh (login node, ~3.9 GB).
        CHASE_NAME=${DATASET#chase_}
        case "$CHASE_NAME" in
            dblp_s)    CHASE_PKL=DBLP-S.pkl;    CHASE_TOPK=${CHASE_TOPK:-0} ;;
            gossipcop) CHASE_PKL=GossipCop.pkl; CHASE_TOPK=${CHASE_TOPK:-0} ;;
            music)     CHASE_PKL=Music.pkl;     CHASE_TOPK=${CHASE_TOPK:-0} ;;
        esac
        if [[ ! -f "dataset/chase/$CHASE_PKL" ]]; then
            echo "[error] dataset/chase/$CHASE_PKL not found;" \
                 "run: bash scripts/dblp/fetch_chase.sh (login node)" >&2
            exit 3
        fi
        CHASE_FEATS=${CHASE_FEATS:-1}
        CHASE_NORM=${CHASE_NORM:-1}
        CHASE_PCA=${CHASE_PCA:-0}
        EDGE_FEATS=0
        # gossipcop (6091 nodes, 776-d features): BATCH 4 like the CLDG
        # graphs. At 8 the encoder-training DyG-JEPA arms (naive / ema /
        # vicreg) OOM a 32 GB V100. dblp_s at full size (34,717 nodes) is
        # ~5.7x more adjacency cells per snapshot than gossipcop: BATCH 1.
        case "$CHASE_NAME" in
            dblp_s)    BATCH=${BATCH:-1} ;;
            gossipcop) BATCH=${BATCH:-4} ;;
        esac
        DATA_ARG="--data_source $DATASET --chase_data_dir dataset/chase \
--chase_top_k $CHASE_TOPK --chase_feat_pca $CHASE_PCA"
        [[ "$CHASE_FEATS" == "0" ]] && DATA_ARG="$DATA_ARG --no_chase_node_features"
        [[ "$CHASE_NORM" == "0" ]] && DATA_ARG="$DATA_ARG --no_chase_feat_norm"
        ;;
    cldg_dblp|cldg_patent)
        # CLDG-release graphs with STATIC per-node classes (dblp: 10, patent:
        # 6) -- the static node-classification task, probed with
        # `--task multiclass --label_source dataset` (node-level split).
        # Files come from scripts/dblp/extract_cldg_datasets.sh.
        #   CLDG_BINS   dblp 0  = one snapshot per distinct timestamp: the 27
        #                        yearly co-authorship graphs the release has.
        #               patent 30 equal-FREQUENCY bins (~1.4k edges each): the
        #                        axis is so skewed that equal-width bins put
        #                        2 edges in the first snapshot and 15k in the
        #                        29th.
        #   CLDG_TOPK   8000 most active nodes for both. The dense (T, N, N)
        #               snapshots make 28k-node DBLP infeasible (21 GB per
        #               timeline); 8000 keeps 48% of DBLP's edges (min degree
        #               16) and 90% of patent's (only degree-1 leaves go).
        #   CLDG_FEATS  1 = append the 128-dim feature.txt rows (CLDG's own
        #               encoder input, a sinusoidal node-index code). 0 for
        #               the topology-only ablation.
        # No edge features exist, so EDGE_FEATS is forced off for every task.
        CLDG_NAME=${DATASET#cldg_}
        if [[ ! -f "dataset/cldg/${CLDG_NAME}/${CLDG_NAME}.txt" ]]; then
            echo "[error] dataset/cldg/${CLDG_NAME}/${CLDG_NAME}.txt not found;" \
                 "run: bash scripts/dblp/extract_cldg_datasets.sh" >&2
            exit 3
        fi
        case "$CLDG_NAME" in
            dblp)   CLDG_BINS=${CLDG_BINS:-0} ;;
            patent) CLDG_BINS=${CLDG_BINS:-30} ;;
        esac
        CLDG_TOPK=${CLDG_TOPK:-8000}
        CLDG_FEATS=${CLDG_FEATS:-1}
        CLDG_BINNING=${CLDG_BINNING:-equal_freq}
        EDGE_FEATS=0
        # ~8000 dense nodes per window: BATCH 4 is what fits an 11 GB card
        # (the normalised adjacency alone is B x 4 x N^2 floats).
        BATCH=${BATCH:-4}
        DATA_ARG="--data_source $DATASET --cldg_data_dir dataset/cldg \
--cldg_num_bins $CLDG_BINS --cldg_binning $CLDG_BINNING --cldg_top_k $CLDG_TOPK"
        [[ "$CLDG_FEATS" == "0" ]] && DATA_ARG="$DATA_ARG --no_cldg_file_features"
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
--synth_persistence ${SYN_PERSISTENCE:-0.6} --synth_diffusion ${SYN_DIFFUSION:-0.3} \
--synth_feature_readout ${SYN_READOUT:-tanh}"
        ;;
    *)
        echo "[error] unsupported DATASET='$DATASET'. Supported: dblp, " \
             "jodie_{mooc,reddit,wikipedia}, reddit_{body,title}, " \
             "cldg_{dblp,patent} (alias patent), tgbn_{trade,genre,reddit,token}, bts_flights, nyc_taxi, " \
             "chase_{dblp_s,gossipcop,music}, " \
             "england_covid, metr_la, " \
             "synthetic, synthetic_dyn" >&2
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
