"""Factory for the DTDG datasets used in this project.

Centralises the `--data_source` dispatch so each of the four entry-point
scripts (pretrain, train_cjepa, diagnose, ablations) gets a single call
site instead of branching on data source.

Data sources:
  - "dblp":            DBLPDynamicGraphDataset (real CSV at data_path)
  - "jodie_reddit":    JODIE Reddit binned to DTDG
  - "jodie_wikipedia": JODIE Wikipedia binned to DTDG
  - "jodie_mooc":      JODIE MOOC binned to DTDG (densest state labels)
  - "cldg_dblp":       CLDG-format DBLP (27 yearly snapshots, 10 static classes)
  - "cldg_patent":     CLDG-format patent citations (6 static classes)
  - "tgbn_trade":      TGB tgbn-trade (31 yearly snapshots, per-(year, nation)
                       export-share label vectors, NDCG@10 ranking probe)
  - "bts_flights":     BTS on-time flights: daily airport network, label =
                       airport in a delayed state (recurring), binary probe
  - "nyc_taxi":        NYC TLC taxi trips: hourly zone-to-zone network, label =
                       zone congested (low trip speed; recurring and periodic),
                       binary probe
  - "tgbn_genre" / "tgbn_reddit" / "tgbn_token":
                       TGB weekly user->class interaction streams; label =
                       the user's next-week class distribution, same probe
  - "chase_dblp_s" / "chase_gossipcop" / "chase_music":
                       CHASE (Kim et al., NeurIPS'26) snapshot graphs with
                       PER-SNAPSHOT node labels (research area / fake-news
                       consumption / dominant genre) and per-snapshot node
                       features; binary probe for the first two, dynamic
                       multiclass (5 classes) for music

JODIE-specific kwargs (num_bins, jodie_data_path; the full dataset is always used)
are passed through; the DBLP class quietly ignores them.
"""
from __future__ import annotations

from pathlib import Path

from src.custom_codes.cldg_format_dataset import CLDGFormatDynamicGraphDataset
from src.custom_codes.dblp_dataset import DBLPDynamicGraphDataset
from src.custom_codes.ehrshot_dataset import EHRshotPerPatientDataset
from src.custom_codes.england_covid_dataset import (
    ENGLAND_COVID_URL,
    EnglandCovidDataset,
)
from src.custom_codes.jodie_dataset import JODIEDynamicGraphDataset
from src.custom_codes.metr_la_dataset import MetrLADataset
from src.custom_codes.reddit_dataset import RedditHyperlinkDynamicGraphDataset
from src.custom_codes.synthetic_dyn_dataset import SyntheticDynamicGraphDataset
from src.custom_codes.bts_flights_dataset import BTSFlightsDataset
from src.custom_codes.chase_dataset import CHASEDataset
from src.custom_codes.nyc_taxi_dataset import NYCTaxiDataset
from src.custom_codes.tgbn_dataset import TGBNInteractionDataset, TGBNTradeDataset


def add_dataset_cli_flags(p) -> None:
    """Attach the `--data_source` family of flags to an ArgumentParser.

    Keeps the four scripts in sync — adding a new dataset only touches this
    function and the dispatch below."""
    p.add_argument("--data_source", default="dblp",
                   choices=["dblp", "jodie_reddit", "jodie_wikipedia",
                            "jodie_mooc", "reddit_body", "reddit_title",
                            "ehrshot", "metr_la", "england_covid",
                            "synthetic_dyn", "cldg_dblp", "cldg_patent",
                            "tgbn_trade", "tgbn_genre", "tgbn_reddit",
                            "tgbn_token", "bts_flights", "nyc_taxi",
                            "chase_dblp_s", "chase_gossipcop", "chase_music"],
                   help="Which DTDG to load. 'dblp' covers the real DBLP "
                        "edges CSV (unlabelled, link prediction). 'cldg_dblp' "
                        "/ 'cldg_patent' load the CLDG-release graphs under "
                        "dataset/cldg/ (static per-node classes; see "
                        "--cldg_* and src/custom_codes/cldg_format_dataset.py). "
                        "'jodie_*' load Stanford's JODIE bipartite CSVs and "
                        "bin them into DTDG snapshots. 'reddit_body' / "
                        "'reddit_title' load Kumar 2018's homogeneous "
                        "subreddit-hyperlink graphs (SNAP TSVs). 'ehrshot' "
                        "loads the per-patient EHR temporal event graphs "
                        "preprocessed by scripts/dblp/preprocess_ehrshot.py. "
                        "'synthetic_dyn' is the latent-driven generator used to "
                        "test latent- vs observation-space prediction; see "
                        "--synth_* below and src/custom_codes/"
                        "synthetic_dyn_dataset.py.")
    p.add_argument("--jodie_data_path", default=None,
                   help="Path to a JODIE CSV (Stanford format). Defaults to "
                        "dataset/jodie/<dataset>.csv when --data_source is jodie_*.")
    p.add_argument("--jodie_num_bins", type=int, default=30,
                   help="Number of DTDG snapshots to bin JODIE events into.")
    p.add_argument("--reddit_data_path", default=None,
                   help="Path to a SNAP Reddit-hyperlinks TSV. Defaults to "
                        "dataset/reddit/soc-redditHyperlinks-<variant>.tsv "
                        "when --data_source is reddit_*.")
    p.add_argument("--reddit_num_bins", type=int, default=36,
                   help="Number of DTDG snapshots (default 36 = monthly over "
                        "the ~3-year span of the SNAP data).")
    p.add_argument("--reddit_top_k", type=int, default=1000,
                   help="Keep the top-K most-active subreddits (combined "
                        "src+tgt count).")
    p.add_argument("--with_edge_features", action="store_true",
                   help="For JODIE / Reddit-Body, additionally load the "
                        "per-edge LIWC features and aggregate them to "
                        "per-(bin, node) means. These get appended to the "
                        "8 graph-derived features, so the encoder sees both "
                        "topology stats AND semantic content per node. "
                        "Required for feature-mask pretraining; ignored on "
                        "data sources without edge features (dblp, sbm).")
    p.add_argument("--no_normalize_edge_features",
                   dest="normalize_edge_features", action="store_false",
                   help="Disable per-dim z-scoring of aggregated edge features. "
                        "Default is to normalise — unnormalised LIWC / "
                        "PROPERTIES have per-dim stds spanning many orders of "
                        "magnitude, which gives the MSE pretrain loss a huge "
                        "dynamic range and produces bimodal multi-seed results. "
                        "Only disable for ablation studies.")
    p.add_argument("--ehrshot_data_path", default="dataset/ehrshot/ehrshot_cohort.pkl",
                   help="Path to the preprocessed EHRshot pickle (from "
                        "scripts/dblp/preprocess_ehrshot.py).")
    p.add_argument("--ehrshot_max_visits", type=int, default=60,
                   help="Truncate each patient to their last K visits. "
                        "0 = keep all. Caps GPU memory growth on tail "
                        "patients with 600+ outpatient visits.")
    p.add_argument("--ehrshot_max_windows", type=int, default=16,
                   help="Cap windows per patient (subsampled uniformly for "
                        "train; aligned to recent for val/test). 0 = uncapped.")
    p.add_argument("--metr_la_data_dir", default="dataset/metr_la",
                   help="METR-LA data dir (metr-la.h5 + adj_mx.pkl, or HF "
                        "parquet).")
    p.add_argument("--metr_la_stride", type=int, default=1,
                   help="Step between consecutive METR-LA windows. 1 (default) "
                        "is the DCRNN convention and what the FINAL table must "
                        "use. Raise it only for hyperparameter search or "
                        "diagnostics: the stride-1 windows overlap by 23/24, so "
                        "e.g. --metr_la_stride 12 gives non-overlapping history "
                        "and cuts a run ~12x with essentially the same ranking "
                        "signal. Never mix strides within one results table.")
    p.add_argument("--metr_la_synthetic", action="store_true",
                   help="Use the 50-node synthetic traffic graph instead of the "
                        "real METR-LA files (wiring/shape checks only).")
    p.add_argument("--england_covid_data_dir", default="dataset/england_covid",
                   help="Folder for the cached england_covid.json (the file "
                        "PyG Temporal uses). Auto-downloaded on first use.")
    p.add_argument("--england_covid_url", default=ENGLAND_COVID_URL,
                   help="Source URL for england_covid.json (only used if the "
                        "cached copy is absent).")
    p.add_argument("--england_covid_use_pyg", action="store_true",
                   help="Load via torch_geometric_temporal's "
                        "EnglandCovidDatasetLoader instead of the vendored "
                        "JSON cache (requires that package installed).")
    # --- CLDG-format (dblp / patent) knobs ----------------------------------
    p.add_argument("--cldg_data_dir", default="dataset/cldg",
                   help="Folder holding <name>/<name>.txt, node2label.txt and "
                        "feature.txt (from scripts/dblp/extract_cldg_datasets.sh).")
    p.add_argument("--cldg_num_bins", type=int, default=30,
                   help="Number of DTDG snapshots. 0 = one snapshot per "
                        "DISTINCT timestamp, the dataset's own granularity "
                        "(dblp: 27 years). patent has 891 distinct values on "
                        "a very skewed axis, so it is binned.")
    p.add_argument("--cldg_binning", choices=["equal_width", "equal_freq"],
                   default="equal_freq",
                   help="How --cldg_num_bins > 0 cuts the time axis. "
                        "'equal_freq' (quantile bins, ~equal edge counts) is "
                        "the only usable choice on patent, where 83%% of the "
                        "edges fall in the last tenth of the axis.")
    p.add_argument("--cldg_top_k", type=int, default=0,
                   help="Keep the K most active nodes (combined src+dst "
                        "count); 0 = every node. The dense (T, N, N) snapshots "
                        "make the 28k-node DBLP graph infeasible at full "
                        "size; labels and features follow the subset.")
    p.add_argument("--no_cldg_file_features", dest="cldg_file_features",
                   action="store_false",
                   help="Do NOT append the 128-dim feature.txt rows (the "
                        "sinusoidal node-index code CLDG trains on) to the "
                        "8 graph-derived features. Ablation only: without an "
                        "identity code the frozen latent carries little about "
                        "a node's class.")
    p.set_defaults(cldg_file_features=True)
    # --- TGB node property prediction (tgbn-trade) ---------------------------
    p.add_argument("--tgbn_data_dir", default=None,
                   help="Folder with tgbn-<name>_edgelist.csv and "
                        "tgbn-<name>_node_labels.csv (scripts/dblp/"
                        "fetch_tgbn.sh <name>). Default dataset/tgb/tgbn-<name>. "
                        "trade: snapshots are the native years and the label "
                        "is each nation's export-share vector. genre / reddit / "
                        "token: weekly bipartite user->class snapshots, label = "
                        "the user's next-week class distribution. All probed "
                        "with run_block_probe.py --task ranking (NDCG@10).")
    # --- BTS airport delay propagation --------------------------------------
    p.add_argument("--bts_data_dir", default="dataset/bts",
                   help="Folder with the BTS on-time monthly zips "
                        "(scripts/dblp/fetch_bts_flights.sh <year>).")
    p.add_argument("--bts_years", type=int, nargs="+", default=[2019],
                   help="Calendar years to load (consecutive). One year = "
                        "365 daily snapshots of ~360 airports.")
    p.add_argument("--bts_bin_hours", type=int, default=24,
                   help="Snapshot length in hours (must divide 24). 24 = one "
                        "per day (FlightDate); shorter bins use the scheduled "
                        "LOCAL departure hour.")
    p.add_argument("--bts_label", choices=["share_del15", "mean_delay"],
                   default="share_del15",
                   help="Delay state: share of departures delayed >= 15 min "
                        "above --bts_threshold (default 0.30), or mean "
                        "departure delay in minutes above it (use ~15).")
    p.add_argument("--bts_threshold", type=float, default=0.30)
    p.add_argument("--bts_min_departures", type=int, default=5,
                   help="Airport-bins with fewer flown departures get no label.")
    p.add_argument("--no_bts_ops_features", dest="bts_ops_features",
                   action="store_false",
                   help="Drop the operational descriptors (departures, "
                        "arrivals, delay shares, cancellation rate, "
                        "day-of-week) and keep only the 8 graph statistics.")
    p.set_defaults(bts_ops_features=True)
    # --- NYC taxi zone congestion -------------------------------------------
    p.add_argument("--nyc_data_dir", default="dataset/nyc_taxi",
                   help="Folder with the TLC monthly parquet files "
                        "(scripts/dblp/fetch_nyc_taxi.sh <YYYY-MM> ...).")
    p.add_argument("--nyc_months", nargs="+",
                   default=["2019-01", "2019-02", "2019-03"],
                   help="Months to load, YYYY-MM (consecutive). One month = "
                        "~720 hourly snapshots of ~260 taxi zones.")
    p.add_argument("--nyc_bin_hours", type=int, default=1,
                   help="Snapshot length in hours on the pickup time (must "
                        "divide 24).")
    p.add_argument("--nyc_kind", choices=["yellow", "green"], default="yellow")
    p.add_argument("--nyc_label", choices=["rel_speed", "abs_speed"],
                   default="rel_speed",
                   help="Congestion state: mean pickup-trip speed below the "
                        "zone's own --nyc_threshold quantile over the train "
                        "bins (default 0.25), or below --nyc_threshold mph "
                        "(use ~10).")
    p.add_argument("--nyc_threshold", type=float, default=0.25)
    p.add_argument("--nyc_min_trips", type=int, default=10,
                   help="Zone-bins with fewer valid-speed pickups get no label.")
    p.add_argument("--nyc_min_edge_trips", type=int, default=1,
                   help="Trips needed in a bin for a pickup->dropoff edge.")
    p.add_argument("--no_nyc_ops_features", dest="nyc_ops_features",
                   action="store_false",
                   help="Drop the trip descriptors (pickups, dropoffs, speed, "
                        "trip length, day-of-week / hour) and keep only the "
                        "8 graph statistics.")
    p.set_defaults(nyc_ops_features=True)
    # --- CHASE dynamic-label benchmarks (dblp_s / gossipcop / music) --------
    p.add_argument("--chase_data_dir", default="dataset/chase",
                   help="Folder with DBLP-S.pkl / GossipCop.pkl / Music.pkl "
                        "(scripts/dblp/fetch_chase.sh).")
    p.add_argument("--chase_top_k", type=int, default=0,
                   help="Keep the K most active nodes (sum of degrees over "
                        "the timeline); 0 = every node. The dense (T, N, N) "
                        "snapshots make 34,717-node DBLP-S infeasible at "
                        "full size; labels and features follow the subset.")
    p.add_argument("--no_chase_node_features", dest="chase_node_features",
                   action="store_false",
                   help="Do NOT append the release's per-snapshot node "
                        "vectors (768-dim BERT / 779-dim listening "
                        "descriptors) to the 8 graph statistics. Ablation.")
    p.set_defaults(chase_node_features=True)
    p.add_argument("--no_chase_feat_norm", dest="chase_feat_norm",
                   action="store_false",
                   help="Skip the per-dimension z-scoring of the node vectors "
                        "(fitted on the train snapshots' active rows).")
    p.set_defaults(chase_feat_norm=True)
    p.add_argument("--chase_feat_pca", type=int, default=0,
                   help="Reduce the node vectors to this many PCA dims "
                        "(fitted on the train snapshots); 0 = keep all.")
    p.add_argument("--tgbn_bin_days", type=int, default=7,
                   help="genre / reddit / token only: snapshot length in days. "
                        "7 = TGB's weekly label grid (label(b) == snapshot b's "
                        "interaction distribution); coarser bins keep the label "
                        "queried at the bin start, which still spans one week.")
    p.add_argument("--tgbn_max_bins", type=int, default=0,
                   help="genre / reddit / token only: keep only the LAST k "
                        "labelled bins (plus the lead-in) and the nodes seen in "
                        "them -- a time cut, not a node subset; every label row "
                        "of the kept weeks survives. 0 = whole stream. reddit "
                        "(~730 weeks, 11.8k nodes) needs this for the dense "
                        "pipeline; genre does not.")
    p.add_argument("--no_tgbn_weight_features", dest="tgbn_weight_features",
                   action="store_false",
                   help="Drop the 5 per-(year, nation) trade-weight "
                        "descriptors (max share, entropy, partner counts, "
                        "import mass) that carry the edge weights the bool "
                        "snapshot loses.")
    p.set_defaults(tgbn_weight_features=True)
    # --- synthetic_dyn generator knobs -------------------------------------
    # The experiment sweeps --synth_node_noise (observation noise, which the
    # model cannot predict) while holding --synth_ou_sigma LOW (so the latent
    # CAUSE stays predictable). Raising both at once cancels the effect out.
    p.add_argument("--synth_nodes", type=int, default=200)
    p.add_argument("--synth_steps", type=int, default=300)
    p.add_argument("--synth_feat_dim", type=int, default=16)
    p.add_argument("--synth_communities", type=int, default=4)
    p.add_argument("--synth_node_noise", type=float, default=0.10,
                   help="Observation noise added AFTER the latent readout. The "
                        "sweep axis: higher = lower SNR with an unchanged "
                        "underlying cause, which is where latent-space "
                        "prediction should pull ahead of input-space.")
    p.add_argument("--synth_target_noise", type=float, default=0.0,
                   help="Observation noise on the TARGET's history, in units of "
                        "the target's train std. -1 = match the feature SNR "
                        "automatically. Without it the persistence baseline "
                        "reads the clean target and is immune to "
                        "--synth_node_noise, so the sweep only handicaps the "
                        "learned arms and headroom decays to ~0 (measured: "
                        "12.5%% -> 0.3%% over node_noise 0 -> 1.0) instead of "
                        "growing (29%% -> 58%%). Evaluation is always against "
                        "the CLEAN future value.")
    p.add_argument("--synth_ou_sigma", type=float, default=0.25,
                   help="LATENT noise. Raising this makes the cause itself "
                        "unpredictable and hurts every method — keep it low "
                        "when sweeping --synth_node_noise.")
    p.add_argument("--synth_ou_theta", type=float, default=0.15,
                   help="Latent mean reversion; higher = more predictable.")
    p.add_argument("--synth_momentum", type=float, default=0.8,
                   help="Velocity autocorrelation of the latent. 0 = pure OU, "
                        "which is Markov-1: history adds nothing and the oracle "
                        "beats copy-last by only ~2%%, so every comparison "
                        "returns a null. >0 makes the trajectory predictable "
                        "several steps out (0.8 -> ~28%% headroom) and makes "
                        "inferring velocity from noisy observations the thing "
                        "being tested.")
    p.add_argument("--synth_switch_prob", type=float, default=0.03,
                   help="Base community-switch rate. Drives how many positives "
                        "the binary task has; too low and AUROC is noise.")
    p.add_argument("--synth_switch_mode", choices=["iid", "latent_driven"],
                   default="latent_driven",
                   help="'latent_driven' makes switches anticipatable from the "
                        "history (oracle AUROC ~0.65). 'iid' makes them "
                        "unpredictable, so copy-last is optimal — a "
                        "persistence-dominated control, not a real task.")
    p.add_argument("--synth_anchor_scale", type=float, default=2.0,
                   help="Spread of the community anchors in latent space. "
                        "Closer anchors (smaller) = more boundary crossings "
                        "under nearest_anchor labels = higher state switch rate.")
    p.add_argument("--synth_label_mode", choices=["hazard", "nearest_anchor"],
                   default="hazard",
                   help="What a community IS. 'hazard' (legacy) = a separately "
                        "sampled Markov label with a uniform destination, so "
                        "which community comes next is a coin flip. "
                        "'nearest_anchor' = the anchor the latent is closest "
                        "to: the label is a readout of z, switches are boundary "
                        "crossings, and the state at t+1 is knowable from the "
                        "trajectory. Required for the node-state task.")
    p.add_argument("--synth_hysteresis", type=float, default=0.0,
                   help="nearest_anchor only: relative margin the nearest anchor "
                        "must win by before a node switches (stops boundary "
                        "flicker under momentum). 0 = plain nearest anchor.")
    p.add_argument("--synth_state_target", choices=["switch", "community"],
                   default="switch",
                   help="Which per-(step,node) label ds.node_labels_dyn carries. "
                        "'switch' = binary switch event (rare; AUROC task). "
                        "'community' = the community id (dense multiclass "
                        "node-STATE task; probe with run_block_probe.py "
                        "--task multiclass).")
    p.add_argument("--synth_beta", type=float, default=1.0,
                   help="Edge-logit sharpness on latent similarity. Higher = "
                        "topology more informative about the latent (and about "
                        "community); 0 = edges carry no latent information.")
    p.add_argument("--synth_d_latent", type=int, default=8,
                   help="Latent dimension. Every observable is a readout of "
                        "this vector.")
    p.add_argument("--synth_density", type=float, default=0.05)
    p.add_argument("--synth_persistence", type=float, default=0.6)
    p.add_argument("--synth_diffusion", type=float, default=0.3)
    p.add_argument("--synth_feature_readout", choices=["tanh", "linear"], default="tanh",
                   help="synthetic_dyn node-feature readout of the latent: legacy 2-layer "
                        "tanh, or linear (position/velocity linearly recoverable from "
                        "feature differences; used by the horizon showcase)")
    p.set_defaults(normalize_edge_features=True)
    p.add_argument("--split_mode", choices=["chronological", "iid"],
                   default="chronological",
                   help="Train/val/test split over the shared timeline. "
                        "'chronological' (default) = test strictly later than "
                        "train (the temporal-OOD setting). 'iid' = the same "
                        "non-overlapping windows randomly partitioned (no "
                        "temporal shift; the control). Compare a method's "
                        "margin across the two to isolate temporal-OOD "
                        "robustness.")
    p.add_argument("--with_state_labels", action="store_true",
                   help="JODIE-only: parse the state_label column and expose "
                        "per-user binary labels on ds.node_labels (1 = user "
                        "ever underwent a state change, 0 = otherwise, -1 = "
                        "outside the top-K subset). Use --label_source dataset "
                        "in run_block_probe.py to probe these.")
    p.add_argument("--jodie_state_label_mode",
                   choices=["ever", "event", "state"], default="ever",
                   help="JODIE-only: how to build state-change labels when "
                        "--with_state_labels is set. 'ever' = the legacy static "
                        "per-user 'ever banned' bit (ds.node_labels). 'event' / "
                        "'state' additionally build a per-(bin, node) DYNAMIC "
                        "matrix on ds.node_labels_dyn — 'event' marks the exact "
                        "snapshot a ban occurs (faithful, sparse), 'state' marks "
                        "every snapshot at/after the ban (cumulative, denser). "
                        "The dynamic label is the task the JEPA forecast "
                        "representation is actually built for.")


def build_dynamic_dataset(args, split: str):
    """Build the dataset for ``args``; with ``args.coarse_window = w > 1``
    (the coarse level of a two-level run, src/world_models/hier.py) the
    window lengths in ``args`` are COARSE steps: the underlying dataset is
    built with ``w`` times as many fine snapshots and every window is reduced
    on the fly."""
    w = int(getattr(args, "coarse_window", 1) or 1)
    if w <= 1:
        return _build_dynamic_dataset(args, split)
    import copy
    from src.world_models.hier import CoarseWindowDataset
    fine = copy.copy(args)
    fine.coarse_window = 1
    fine.history_size = args.history_size * w
    fine.num_preds = args.num_preds * w
    return CoarseWindowDataset(_build_dynamic_dataset(fine, split), w,
                               getattr(args, "coarse_mode", "last"))


def _build_dynamic_dataset(args, split: str):
    """Build the train/val/test dataset matching `args.data_source`.

    `args` must already carry: data_source, history_size, num_preds, seed,
    and either the DBLP fields (data_path, synthetic, synthetic_*) or the
    JODIE fields (jodie_*). `split` is "train" | "val" | "test".
    """
    seed = getattr(args, "seed", 42)
    if args.data_source == "dblp":
        return DBLPDynamicGraphDataset(
            data_path=args.data_path,
            split=split,
            history_size=args.history_size,
            num_preds=args.num_preds,
            seed=seed,
        )
    with_edge_features = getattr(args, "with_edge_features", False)
    normalize_edge_features = getattr(args, "normalize_edge_features", True)
    with_state_labels = getattr(args, "with_state_labels", False)
    split_mode = getattr(args, "split_mode", "chronological")
    if args.data_source in ("jodie_reddit", "jodie_wikipedia", "jodie_mooc"):
        dataset = args.data_source.split("_", 1)[1]
        return JODIEDynamicGraphDataset(
            data_path=args.jodie_data_path,
            dataset=dataset,
            split=split,
            history_size=args.history_size,
            num_preds=args.num_preds,
            num_bins=args.jodie_num_bins,
            seed=seed,
            with_edge_features=with_edge_features,
            normalize_edge_features=normalize_edge_features,
            with_state_labels=with_state_labels,
            state_label_mode=getattr(args, "jodie_state_label_mode", "ever"),
            split_mode=split_mode,
        )
    if args.data_source == "ehrshot":
        return EHRshotPerPatientDataset(
            data_path=args.ehrshot_data_path,
            split=split,
            history_size=args.history_size,
            num_preds=args.num_preds,
            max_visits_per_patient=getattr(args, "ehrshot_max_visits", 60),
            max_windows_per_patient=getattr(args, "ehrshot_max_windows", 16),
            seed=seed,
        )
    if args.data_source in ("reddit_body", "reddit_title"):
        variant = args.data_source.split("_", 1)[1]
        return RedditHyperlinkDynamicGraphDataset(
            data_path=args.reddit_data_path,
            variant=variant,
            split=split,
            history_size=args.history_size,
            num_preds=args.num_preds,
            num_bins=args.reddit_num_bins,
            top_k=args.reddit_top_k,
            seed=seed,
            with_edge_features=with_edge_features,
            normalize_edge_features=normalize_edge_features,
        )
    if args.data_source in ("cldg_dblp", "cldg_patent"):
        return CLDGFormatDynamicGraphDataset(
            name=args.data_source.split("_", 1)[1],
            data_dir=getattr(args, "cldg_data_dir", "dataset/cldg"),
            split=split,
            history_size=args.history_size,
            num_preds=args.num_preds,
            num_bins=getattr(args, "cldg_num_bins", 30),
            binning=getattr(args, "cldg_binning", "equal_freq"),
            top_k=getattr(args, "cldg_top_k", 0),
            with_file_features=getattr(args, "cldg_file_features", True),
            seed=seed,
            split_mode=split_mode,
        )
    if args.data_source == "bts_flights":
        return BTSFlightsDataset(
            data_dir=getattr(args, "bts_data_dir", "dataset/bts"),
            years=getattr(args, "bts_years", [2019]),
            split=split,
            history_size=args.history_size,
            num_preds=args.num_preds,
            bin_hours=getattr(args, "bts_bin_hours", 24),
            label_kind=getattr(args, "bts_label", "share_del15"),
            threshold=getattr(args, "bts_threshold", 0.30),
            min_departures=getattr(args, "bts_min_departures", 5),
            with_ops_features=getattr(args, "bts_ops_features", True),
            seed=seed,
            split_mode=split_mode,
        )
    if args.data_source == "nyc_taxi":
        return NYCTaxiDataset(
            data_dir=getattr(args, "nyc_data_dir", "dataset/nyc_taxi"),
            months=getattr(args, "nyc_months", ["2019-01", "2019-02", "2019-03"]),
            split=split,
            history_size=args.history_size,
            num_preds=args.num_preds,
            bin_hours=getattr(args, "nyc_bin_hours", 1),
            kind=getattr(args, "nyc_kind", "yellow"),
            label_kind=getattr(args, "nyc_label", "rel_speed"),
            threshold=getattr(args, "nyc_threshold", 0.25),
            min_trips=getattr(args, "nyc_min_trips", 10),
            min_edge_trips=getattr(args, "nyc_min_edge_trips", 1),
            with_ops_features=getattr(args, "nyc_ops_features", True),
            seed=seed,
            split_mode=split_mode,
        )
    if args.data_source in ("chase_dblp_s", "chase_gossipcop", "chase_music"):
        return CHASEDataset(
            name=args.data_source.split("_", 1)[1],
            data_dir=getattr(args, "chase_data_dir", "dataset/chase"),
            split=split,
            history_size=args.history_size,
            num_preds=args.num_preds,
            top_k=getattr(args, "chase_top_k", 0),
            with_node_features=getattr(args, "chase_node_features", True),
            feat_norm=getattr(args, "chase_feat_norm", True),
            feat_pca=getattr(args, "chase_feat_pca", 0),
            seed=seed,
            split_mode=split_mode,
        )
    if args.data_source in ("tgbn_genre", "tgbn_reddit", "tgbn_token"):
        name = args.data_source.split("_", 1)[1]
        return TGBNInteractionDataset(
            name=name,
            data_dir=getattr(args, "tgbn_data_dir", None) or f"dataset/tgb/tgbn-{name}",
            split=split,
            history_size=args.history_size,
            num_preds=args.num_preds,
            bin_days=getattr(args, "tgbn_bin_days", 7),
            max_bins=getattr(args, "tgbn_max_bins", 0),
            with_weight_features=getattr(args, "tgbn_weight_features", True),
            seed=seed,
            split_mode=split_mode,
        )
    if args.data_source == "tgbn_trade":
        return TGBNTradeDataset(
            data_dir=getattr(args, "tgbn_data_dir", None) or "dataset/tgb/tgbn-trade",
            split=split,
            history_size=args.history_size,
            num_preds=args.num_preds,
            with_weight_features=getattr(args, "tgbn_weight_features", True),
            seed=seed,
            split_mode=split_mode,
        )
    if args.data_source == "england_covid":
        return EnglandCovidDataset(
            data_dir=getattr(args, "england_covid_data_dir",
                             "dataset/england_covid"),
            split=split,
            history_size=args.history_size,
            num_preds=args.num_preds,
            url=getattr(args, "england_covid_url", ENGLAND_COVID_URL),
            use_pyg=getattr(args, "england_covid_use_pyg", False),
            seed=seed,
            split_mode=split_mode,
        )
    if args.data_source == "synthetic_dyn":
        return SyntheticDynamicGraphDataset(
            split=split,
            history_size=args.history_size,
            num_preds=args.num_preds,
            n_nodes=getattr(args, "synth_nodes", 200),
            n_steps=getattr(args, "synth_steps", 300),
            d_node_feat=getattr(args, "synth_feat_dim", 16),
            n_communities=getattr(args, "synth_communities", 4),
            node_noise=getattr(args, "synth_node_noise", 0.10),
            target_noise=getattr(args, "synth_target_noise", 0.0),
            ou_sigma=getattr(args, "synth_ou_sigma", 0.25),
            ou_theta=getattr(args, "synth_ou_theta", 0.15),
            momentum=getattr(args, "synth_momentum", 0.8),
            switch_prob=getattr(args, "synth_switch_prob", 0.03),
            switch_mode=getattr(args, "synth_switch_mode", "latent_driven"),
            target_density=getattr(args, "synth_density", 0.05),
            persistence=getattr(args, "synth_persistence", 0.6),
            diffusion_alpha=getattr(args, "synth_diffusion", 0.3),
            feature_readout=getattr(args, "synth_feature_readout", "tanh"),
            anchor_scale=getattr(args, "synth_anchor_scale", 2.0),
            beta=getattr(args, "synth_beta", 1.0),
            d_latent=getattr(args, "synth_d_latent", 8),
            label_mode=getattr(args, "synth_label_mode", "hazard"),
            hysteresis=getattr(args, "synth_hysteresis", 0.0),
            state_target=getattr(args, "synth_state_target", "switch"),
            seed=seed,
            split_mode=split_mode,
        )
    if args.data_source == "metr_la":
        data_dir = getattr(args, "metr_la_data_dir", "dataset/metr_la")
        synthetic = getattr(args, "metr_la_synthetic", False)
        # MetrLADataset falls back to a 50-node synthetic graph when it finds
        # neither layout in data_dir. That fallback must be explicit: a missing
        # download would otherwise produce a complete, plausible-looking results
        # table computed on fabricated traffic data.
        if not synthetic:
            d = Path(data_dir).expanduser()
            if not ((d / "train.parquet").exists() or (d / "metr-la.h5").exists()):
                raise FileNotFoundError(
                    f"No METR-LA data in {d}: expected either the HF parquet "
                    f"layout ({d}/train.parquet) or the DCRNN layout "
                    f"({d}/metr-la.h5). Run scripts/metr_la/download_metr_la.sh, "
                    f"or pass --metr_la_synthetic to use the toy graph on "
                    f"purpose.")
        return MetrLADataset(
            data_dir=data_dir,
            split=split,
            history_size=args.history_size,
            num_preds=args.num_preds,
            synthetic=synthetic,
            stride=getattr(args, "metr_la_stride", 1),
        )
    
    raise ValueError(f"unknown --data_source {args.data_source!r}")
