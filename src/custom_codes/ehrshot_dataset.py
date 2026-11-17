"""EHRshot per-patient temporal event graph dataset.

Each PATIENT is a small dynamic graph. Nodes = global vocabulary of top-K
concept codes (frozen at preprocess time, fit on training cohort only).
Snapshots = the patient's visits, sorted chronologically.

The adjacency at visit t is the intersection of (a) a static PMI top-K
co-occurrence skeleton precomputed on the training cohort and (b) the
outer product of the active-code mask at that visit. So edges only appear
for clinically meaningful pairs that are *both* present at the current
visit. Without the PMI skeleton the adjacency would be a full clique over
active codes, making the graph layer redundant with the active mask -- a
GCN over a clique just mean-pools. The skeleton is stored in the
preprocessed pickle under key 'pmi_topk_adj'; legacy pickles without it
fall back to clique adjacency (with a one-time warning).

The dataset slides windows of (history_size + num_preds) consecutive visits
over each patient's full visit sequence; windows are concatenated across
patients via the universal :class:`DTDGDataset` per-entity engine.

Returned dict matches the other DTDG datasets so this drops into the unified
train / pretrain scripts unchanged.

The preprocessor at scripts/dblp/preprocess_ehrshot.py produces the input
pickle. See its docstring for vocabulary and filter details.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from src.custom_codes.dtdg.dataset import DTDGDataset
from src.custom_codes.dtdg.features import (
    build_graph_features,
    concept_static_features as _concept_static_features,  # noqa: F401
)


def _active_mask_from_visits(
    visit_seq: List[tuple], num_nodes: int
) -> np.ndarray:
    """Build (T_p, N) bool active mask from a list of (ts, [code_idx, ...])."""
    T_p = len(visit_seq)
    mask = np.zeros((T_p, num_nodes), dtype=bool)
    for t, (_, codes) in enumerate(visit_seq):
        if codes:
            mask[t, codes] = True
    return mask


def _adj_from_active(active_w: np.ndarray,
                     static_skel: np.ndarray) -> np.ndarray:
    """Reconstruct (T, N, N) adjacency from the active mask and a static
    co-occurrence skeleton.

    adj[t] = static_skel & (active[t] outer active[t]); diagonal already
    zero in the skeleton. Computed on demand (per window) to keep RAM use
    linear in N rather than quadratic in (T, N, N).
    """
    T, N = active_w.shape
    adj = np.zeros((T, N, N), dtype=bool)
    for t in range(T):
        a = active_w[t]
        if not a.any():
            continue
        adj[t] = static_skel & np.outer(a, a)
    return adj


class EHRshotPerPatientDataset(Dataset):
    """EHRshot per-patient temporal event graph, sliding-window split.

    Thin wrapper: parses the preprocessed pickle into per-patient active
    masks + a per-window builder, and delegates the flat window indexing /
    subsampling to :meth:`DTDGDataset.from_per_entity`.

    Args:
      data_path: pickle from scripts/dblp/preprocess_ehrshot.py.
      split:     "train" | "val" | "test", per the preprocessed split map.
      history_size / num_preds: window lengths.
      max_visits_per_patient: truncate to the patient's last K visits; 0 = no cap.
      max_windows_per_patient: cap windows per patient (subsampled); 0 = uncapped.
      feat_dim: dynamic node-feature dim added on top of static concept features.
      with_dynamic_features: append the history-based per-(t, node) graph
        features. Defaults True for parity with the other datasets.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        history_size: int = 3,
        num_preds: int = 1,
        max_visits_per_patient: int = 60,
        max_windows_per_patient: int = 16,
        feat_dim: int = 8,
        stride: int = 1,
        seed: int = 42,
        with_dynamic_features: bool = True,
    ):
        assert split in ("train", "val", "test")
        path = Path(data_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"EHRshot pickle not found at {path}. Run\n"
                f"  python scripts/dblp/preprocess_ehrshot.py "
                f"--out {path}\nfirst.")
        with open(path, "rb") as f:
            payload = pickle.load(f)

        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.window = history_size + num_preds
        self.stride = stride
        self.feat_dim = feat_dim
        self.with_dynamic_features = with_dynamic_features

        self.code_vocab: list = payload["code_vocab"]
        self.num_nodes = len(self.code_vocab)
        self._static_feats = _concept_static_features(
            payload["code_vocab_type"], payload["code_freq"], self.num_nodes,
        )
        if "pmi_topk_adj" in payload:
            self._static_skel = payload["pmi_topk_adj"].astype(bool)
            print(f"[ehrshot] PMI skeleton density: "
                  f"{self._static_skel.sum() / self.num_nodes**2:.4f}, "
                  f"mean_degree={self._static_skel.sum(1).mean():.1f}",
                  flush=True)
        else:
            # Legacy pickle: clique fallback.
            print("[ehrshot] WARNING: pickle has no 'pmi_topk_adj' key -- "
                  "falling back to clique adjacency (graph layer "
                  "redundant with active mask). Re-run preprocess with "
                  "--pmi_topk > 0 to fix.", flush=True)
            self._static_skel = ~np.eye(self.num_nodes, dtype=bool)

        pids = payload["patient_ids"]
        splits = payload["patient_split"]
        visits_by_pid: dict = payload["patient_visits"]
        keep = [pid for pid, s in zip(pids, splits) if s == split]
        if not keep:
            raise ValueError(
                f"No patients in split={split!r}. The preprocessed pickle's "
                f"splits are: {sorted(set(splits))}.")

        self.patient_ids = keep
        self.patient_active: list = []           # list of (T_p, N) bool arrays
        self.patient_visit_ts: list = []         # list of (T_p,) int64 arrays
        for pid in keep:
            seq = visits_by_pid[pid]
            if max_visits_per_patient > 0 and len(seq) > max_visits_per_patient:
                seq = seq[-max_visits_per_patient:]
            T_p = len(seq)
            if T_p < self.window:
                self.patient_active.append(np.zeros((0, self.num_nodes), bool))
                self.patient_visit_ts.append(np.zeros(0, dtype=np.int64))
                continue
            self.patient_active.append(
                _active_mask_from_visits(seq, self.num_nodes))
            self.patient_visit_ts.append(
                np.array([t for t, _ in seq], dtype=np.int64))

        self._dyn_feat_dim = feat_dim if with_dynamic_features else 0
        self.feature_dim_total = (self._static_feats.shape[1]
                                  + self._dyn_feat_dim)
        self.year_offset = 0
        self.node_labels = None

        lengths = [pa.shape[0] for pa in self.patient_active]
        self._ds = DTDGDataset.from_per_entity(
            lengths=lengths,
            window_builder=self._build_window,
            history_size=history_size, num_preds=num_preds,
            num_nodes=self.num_nodes, feat_dim=self.feature_dim_total,
            split=split, stride=stride,
            max_windows_per_entity=max_windows_per_patient,
            node_labels=None, year_offset=0,
        )
        self.starts = self._ds.starts
        if self.starts.size == 0:
            raise ValueError(
                f"Split {split!r} has no valid windows. "
                f"window={self.window}, max_visits_per_patient="
                f"{max_visits_per_patient}, "
                f"max_windows_per_patient={max_windows_per_patient}, "
                f"#patients={len(keep)}.")

        print(f"[ehrshot] split={split}  patients={len(keep)}  "
              f"windows={len(self.starts)}  N={self.num_nodes}  "
              f"feat_dim={self.feature_dim_total}", flush=True)

    def _build_window(self, p_idx: int, t_start: int) -> dict:
        active_full = self.patient_active[p_idx]
        t_end = t_start + self.window
        active_w = active_full[t_start:t_end]                    # (T, N) bool
        adj_w = _adj_from_active(active_w, self._static_skel)     # (T, N, N) bool

        static = self._static_feats[None].repeat(self.window, axis=0)  # (T,N,F0)
        if self.with_dynamic_features:
            dyn = build_graph_features(
                adj_w, active_w, feat_dim=self._dyn_feat_dim,
                seed=p_idx * 10000 + t_start,
            )
            feats = np.concatenate([static, dyn], axis=-1)
        else:
            feats = static

        adj_t = torch.from_numpy(adj_w)
        active_t = torch.from_numpy(active_w)
        feats_t = torch.from_numpy(feats.astype(np.float32))
        return {
            "x_hist": feats_t[: self.history_size],
            "x_fut":  feats_t[self.history_size:],
            "adj":    adj_t,
            "active": active_t,
            "year":   t_start + self.year_offset,
        }

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        return self._ds[idx]

    @property
    def feature_dim(self) -> int:
        return self.feature_dim_total
