"""DTDG loader for the synthetic latent-driven dynamic graph generator.

Wraps ``dynamic_graph_generator.generate_dynamic_graph`` in the project's
canonical ``SnapshotSequence`` -> ``DTDGDataset`` path, so the synthetic data
flows through the same encoder, predictor, probe and split logic as the real
datasets.

The generator draws every observable from one per-node latent ``z``, so
``--synth_node_noise`` adds noise the model cannot predict while leaving the
underlying cause exactly as predictable as before. Sweeping that knob against
``--predict_space {latent,input}`` tests latent-space against
observation-space prediction.

Noise on the features alone degrades every learned method while leaving the
persistence baseline untouched, because persistence reads the clean target's
own history. With ``--synth_target_noise`` the model and the persistence
baseline both read a noisy history and are both scored against the clean
future, so denoising pays. ``-1`` matches the target's SNR to the feature SNR
automatically.

Before trusting any comparison, read the headroom line printed on the test
split: it is the oracle-vs-persistence gap given perfect knowledge of the
latent. Below ~10% the setup cannot resolve a difference between methods.
``--synth_momentum`` is the knob that creates headroom. The oracle is computed
from the clean latents, so it is blind to observation noise; read it together
with the persistence line below it, which is not.

Three targets are exposed:

  regression (recommended)  ``labels_cont``, a smooth scalar readout of z.
      Dense: every (step, node) is a target, so there is no class imbalance.
      Linear extrapolation does worse than copy-last because the latent is
      mean-reverting, so beating persistence requires modelling the dynamics.

  binary                    ``node_labels_dyn`` = 1 at the step a node changes
      community. Only meaningful with ``switch_mode="latent_driven"``, where an
      imminent switch is visible in the history; under "iid" the switches are
      unpredictable by construction. Switch events are rare, so check the
      positive count printed at construction.

  multiclass state          ``node_labels_dyn`` = the community id at every
      step (``state_target="community"``). Only meaningful with
      ``label_mode="nearest_anchor"``, where the label is a readout of the
      latent (see ``state_headroom`` in the generator). Under the hazard label
      the destination is a coin flip and copy-last is optimal.

Features are z-scored with train-split statistics, so raising
``--synth_node_noise`` does not also inflate the input scale.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from torch.utils.data import Dataset

from src.custom_codes.dtdg.canonical import SnapshotSequence
from src.custom_codes.dtdg.dataset import DTDGDataset

# train/val/test are built as three separate objects in one process and must
# see the identical graph, so memoise on the full parameter tuple.
_CACHE: dict = {}


def _generate(params: tuple) -> dict:
    if params not in _CACHE:
        from dynamic_graph_generator import generate_dynamic_graph
        _CACHE[params] = generate_dynamic_graph(**dict(params))
    return _CACHE[params]


class SyntheticDynamicGraphDataset(Dataset):
    """Latent-driven synthetic DTDG with dynamic features and labels."""

    SPLITS: Tuple[float, float, float] = (0.7, 0.1, 0.2)

    def __init__(
        self,
        split: str = "train",
        history_size: int = 3,
        num_preds: int = 1,
        stride: int = 1,
        n_nodes: int = 200,
        n_steps: int = 300,
        d_node_feat: int = 16,
        n_communities: int = 4,
        node_noise: float = 0.10,
        target_noise: float = 0.0,
        ou_sigma: float = 0.25,
        ou_theta: float = 0.15,
        momentum: float = 0.8,
        switch_prob: float = 0.03,
        switch_mode: str = "latent_driven",
        target_density: float = 0.05,
        persistence: float = 0.6,
        diffusion_alpha: float = 0.3,
        feature_readout: str = "tanh",
        anchor_scale: float = 2.0,
        beta: float = 1.0,
        d_latent: int = 8,
        label_mode: str = "hazard",
        hysteresis: float = 0.0,
        state_target: str = "switch",
        seed: int = 42,
        split_mode: str = "chronological",
        verbose: bool = True,
    ):
        assert split in ("train", "val", "test")
        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.window = history_size + num_preds

        params = (
            ("n_nodes", n_nodes), ("n_steps", n_steps),
            ("d_node_feat", d_node_feat), ("n_communities", n_communities),
            ("node_noise", node_noise), ("ou_sigma", ou_sigma),
            ("ou_theta", ou_theta), ("momentum", momentum),
            ("switch_prob", switch_prob),
            ("switch_mode", switch_mode), ("target_density", target_density),
            ("persistence", persistence), ("diffusion_alpha", diffusion_alpha),
            ("feature_readout", feature_readout),
            ("anchor_scale", anchor_scale), ("beta", beta),
            ("d_latent", d_latent), ("label_mode", label_mode),
            ("hysteresis", hysteresis),
            ("seed", seed),
        )
        assert state_target in ("switch", "community"), state_target
        data = _generate(params)

        adj = np.stack(data["snapshots"]).astype(bool)            # (T, N, N)
        feats = data["node_features"].astype(np.float32)          # (T, N, F)
        comm = data["labels"].astype(np.int64)                    # (T, N)
        cont = data["labels_cont"].astype(np.float32)             # (T, N)
        T, N = comm.shape
        # Every node exists at every step in this generator.
        active = np.ones((T, N), dtype=bool)

        train_end = max(1, int(T * self.SPLITS[0]))
        # Z-score features on train steps only, so the noise sweep varies SNR
        # and not the input scale.
        mu = feats[:train_end].reshape(-1, feats.shape[-1]).mean(0)
        sd = feats[:train_end].reshape(-1, feats.shape[-1]).std(0) + 1e-6
        feats = (feats - mu) / sd

        # Regression target, normalised the same way (train stats only).
        self.mean = float(cont[:train_end].mean())
        self.std = float(cont[:train_end].std() + 1e-6)
        self.raw_signal = cont                                    # (T, N) CLEAN
        # Signed, roughly zero-mean target: a `true > 0` metric mask would
        # drop most of it, so downstream evals mask on `active` instead.
        self.signed_target = True

        # Observed history of the target. The model and the persistence
        # baseline read this; both are scored against the clean `raw_signal`.
        self.target_noise_sigma = 0.0
        self.signal_obs = cont
        if target_noise != 0.0:
            if target_noise < 0:
                # Match the target's SNR to the feature SNR: node_noise is in
                # units of the readout, so divide by the clean feature spread.
                clean = _generate(tuple((k, 0.0) if k == "node_noise" else (k, v)
                                        for k, v in params))
                rel = node_noise / (float(clean["node_features"].std()) + 1e-12)
            else:
                rel = target_noise
            self.target_noise_sigma = rel * self.std
            # Seeded off the generator seed so all three splits observe the
            # same noisy series.
            eps = np.random.default_rng(seed + 10_000).normal(size=cont.shape)
            self.signal_obs = (cont + self.target_noise_sigma * eps).astype(np.float32)

        # Two dynamic per-(step, node) targets, selected by state_target:
        #   "switch"     binary, 1 at the step a node changes community
        #                (step 0 has no predecessor, so it is 0).
        #   "community"  multiclass, the community id itself.
        switch = np.zeros((T, N), dtype=np.int64)
        switch[1:] = (comm[1:] != comm[:-1]).astype(np.int64)
        self.switch_labels = switch
        self.n_classes = n_communities
        if state_target == "community":
            self.node_labels_dyn = comm
            self.state_label_mode = f"community:{label_mode}"
        else:
            self.node_labels_dyn = switch
            self.state_label_mode = f"switch:{switch_mode}"
        # Static fallback for the multiclass/Louvain probe path: the community
        # each node spends most of its time in.
        self.node_labels = np.array(
            [np.bincount(comm[:, i], minlength=n_communities).argmax()
             for i in range(N)], dtype=np.int64)

        self.latents = data["latents"]        # ground truth, for latent probes
        # Generator state the probes need to reconstruct the hidden velocity
        # v(t) = z(t) - z(t-1) - theta * (a_{c(t)} - z(t-1)).
        self.anchors = data["anchors"]
        self.ou_theta = float(ou_theta)
        self.momentum = float(momentum)
        self.feature_readout = feature_readout
        self.communities = comm
        self.num_nodes = N
        self.feat_dim = feats.shape[-1]

        seq = SnapshotSequence(adj=adj, active=active, feats=feats,
                               node_labels=self.node_labels)
        self._ds = DTDGDataset.from_global(
            seq, split=split, history_size=history_size, num_preds=num_preds,
            stride=stride, splits=self.SPLITS,
            split_mode=split_mode, split_seed=seed,
        )
        self.starts = self._ds.starts

        if verbose and split == "test":
            tgt = sorted({int(s) + history_size for s in self.starts})
            npos = int(switch[tgt].sum())
            rate = npos / max(len(tgt) * N, 1)
            from dynamic_graph_generator import headroom as _headroom
            hr = _headroom(data, splits=self.SPLITS, history=history_size)
            print(f"[synthetic_dyn] N={N} T={T} F={self.feat_dim} "
                  f"node_noise={node_noise} ou_sigma={ou_sigma} "
                  f"momentum={momentum} switch_mode={switch_mode}", flush=True)
            print(f"[synthetic_dyn] headroom: persist MAE={hr['persist_mae']:.4f} "
                  f"oracle MAE={hr['oracle_mae']:.4f} -> "
                  f"{100*hr['headroom_frac']:.1f}%  [{hr['verdict']}] "
                  f"(clean-latent oracle; blind to observation noise)", flush=True)
            # With target_noise=0 this line is constant across the noise
            # sweep, which means the sweep cannot show anything.
            te0 = int(T * (self.SPLITS[0] + self.SPLITS[1]))
            p_clean = np.abs(cont[te0 + 1:] - cont[te0:-1]).mean()
            p_obs = np.abs(cont[te0 + 1:] - self.signal_obs[te0:-1]).mean()
            print(f"[synthetic_dyn] target obs noise sigma="
                  f"{self.target_noise_sigma:.4f} (target std={self.std:.4f}) -> "
                  f"test persistence MAE {p_clean:.4f} (clean base) / "
                  f"{p_obs:.4f} (observed base)", flush=True)
            if self.target_noise_sigma == 0.0 and node_noise > 0:
                print(f"[synthetic_dyn] WARNING: node_noise={node_noise} but "
                      f"target_noise=0 — the persistence baseline reads the "
                      f"CLEAN target, so this noise level only handicaps the "
                      f"learned arms and total headroom shrinks toward 0. Pass "
                      f"--synth_target_noise -1 to match the feature SNR.",
                      flush=True)
            if hr["headroom_frac"] < 0.10:
                print(f"[synthetic_dyn] WARNING: with this little headroom NO "
                      f"method can demonstrate an advantage — any comparison "
                      f"will return a null that says nothing about the methods. "
                      f"Raise --synth_momentum (0.8 gives ~28%) and/or "
                      f"--synth_ou_theta.", flush=True)
            print(f"[synthetic_dyn] test target steps={len(tgt)}  "
                  f"switch positives={npos} ({100*rate:.2f}%)", flush=True)
            if state_target == "community":
                from dynamic_graph_generator import state_headroom as _sh
                sh = _sh(data, splits=self.SPLITS, history=history_size)
                print(f"[synthetic_dyn] state task ({self.state_label_mode}, "
                      f"K={n_communities}): switch rate={100*sh['switch_rate']:.2f}%  "
                      f"persist acc={sh['persist_acc']:.4f}  oracle acc="
                      f"{sh['oracle_acc']:.4f}  oracle acc on switch rows="
                      f"{sh['switch_acc']:.3f}  -> {100*sh['headroom_frac']:.1f}% "
                      f"error reduction  [{sh['verdict']}]", flush=True)
                if label_mode == "hazard":
                    print(f"[synthetic_dyn] WARNING: state_target=community with "
                          f"label_mode=hazard — the destination community is a "
                          f"uniform draw, so persistence is already optimal and "
                          f"no method can beat copy-last. Use "
                          f"--synth_label_mode nearest_anchor.", flush=True)
            if npos < 100 and state_target == "switch":
                print(f"[synthetic_dyn] WARNING: only {npos} switch positives "
                      f"in test — AUROC on the binary task will be very noisy. "
                      f"Raise --synth_switch_prob / --synth_nodes / "
                      f"--synth_steps, or use --downstream_task regression "
                      f"(dense target, no imbalance).", flush=True)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        item = self._ds[idx]
        start = int(self.starts[idx])
        # Clean signal (scored against) and observed signal (read by the
        # model and persistence). Identical unless --synth_target_noise is set.
        item["signal"] = self.raw_signal[start:start + self.window].copy()
        item["signal_obs"] = self.signal_obs[start:start + self.window].copy()
        return item

    @property
    def feature_dim(self) -> int:
        return self.feat_dim
