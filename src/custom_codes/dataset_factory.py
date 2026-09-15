"""Factory for the DTDG datasets used in this project.

Centralises the `--data_source` dispatch so each of the four entry-point
scripts (pretrain, train_cjepa, diagnose, ablations) gets a single call
site instead of branching on data source.

Data sources:
  - "dblp":            DBLPDynamicGraphDataset (real CSV at data_path)
  - "jodie_reddit":    JODIE Reddit binned to DTDG
  - "jodie_wikipedia": JODIE Wikipedia binned to DTDG
  - "jodie_mooc":      JODIE MOOC binned to DTDG (densest state labels)

JODIE-specific kwargs (num_bins, jodie_data_path; the full dataset is always used)
are passed through; the DBLP class quietly ignores them.
"""
from __future__ import annotations

from pathlib import Path

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


def add_dataset_cli_flags(p) -> None:
    """Attach the `--data_source` family of flags to an ArgumentParser.

    Keeps the four scripts in sync — adding a new dataset only touches this
    function and the dispatch below."""
    p.add_argument("--data_source", default="dblp",
                   choices=["dblp", "jodie_reddit", "jodie_wikipedia",
                            "jodie_mooc", "reddit_body", "reddit_title",
                            "ehrshot", "metr_la", "england_covid",
                            "synthetic_dyn"],
                   help="Which DTDG to load. 'dblp' covers the real DBLP "
                        "edges CSV"
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
