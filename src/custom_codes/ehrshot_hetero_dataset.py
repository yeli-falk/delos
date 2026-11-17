"""EHRshot heterogeneous EHR-graph dataset (per-patient, windowed).

Each item is a PyG HeteroData with two node types and one (typed) edge:

    Nodes:
      'visit'   shape (T, F_v)   F_v = visit features (position embedding + ...)
                                  T   = history_size + num_preds  (always)
      'code'    shape (N, F_c)   F_c = code features (type one-hot + log freq)
                                  N   = global vocabulary size

    Edges:
      ('visit', 'records', 'code')    visit -> code (recorded at this visit)
      ('code',  'rev_records', 'visit') reverse, for bidirectional message passing

Only edges from `history_size` *visible* visits are included in the
HeteroData; the remaining `num_preds` future visits are the JEPA target
(their code-attachments are stripped, the predictor reconstructs the
visit-node latents from history alone).

The full ground-truth (which codes were at which visit, including future)
is stored as a dense (T, N) bool `target_mask` on the visit nodes — PyG
treats this as a per-visit-node feature and batches it cleanly into
(B*T, N). The train loop derives the oracle (target-encoder) edges from
this mask, applying per-item code/visit offsets via `batch[...].ptr`.

The returned dict-of-tensors layout matches PyG's `HeteroData` conventions
so a list of items batches via `Batch.from_data_list(...)` in the collate.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import HeteroData
    _PYG = True
except ImportError as e:                                  # pragma: no cover
    HeteroData = None                                      # type: ignore
    _PYG = False
    _PYG_IMPORT_ERR = e


VOCAB_TYPE_DIM = 9   # see scripts/dblp/preprocess_ehrshot.py VOCAB_TYPE_ID


def _require_pyg() -> None:
    if not _PYG:
        raise ImportError(
            "torch_geometric is required for the hetero EHRshot dataset.\n"
            "Install on the cluster with:\n"
            "  pip install torch_geometric "
            "torch_scatter torch_sparse \\\n"
            "      -f https://data.pyg.org/whl/torch-${TORCH_VERSION}.html\n"
            f"Original error: {_PYG_IMPORT_ERR}")


def _concept_static_features(code_vocab_type: np.ndarray,
                             code_freq: np.ndarray,
                             num_nodes: int) -> np.ndarray:
    """Per-code static features: vocabulary type one-hot + log-frequency.
    Identical to the homogeneous EHRshot dataset; same encoder eats them."""
    feats = np.zeros((num_nodes, VOCAB_TYPE_DIM + 1), dtype=np.float32)
    feats[np.arange(num_nodes), code_vocab_type] = 1.0
    f = np.log1p(code_freq.astype(np.float64))
    f = f / max(f.max(), 1.0)
    feats[:, VOCAB_TYPE_DIM] = f.astype(np.float32)
    return feats


def _visit_position_features(T: int, F_v: int) -> np.ndarray:
    """Sinusoidal position features for the T visits in a window. Same idea
    as the V-JEPA temporal positional embedding so the predictor can tell
    which slot it's reconstructing."""
    pos = np.arange(T)[:, None]
    div = np.power(10000, -np.arange(0, F_v, 2) / F_v)
    out = np.zeros((T, F_v), dtype=np.float32)
    out[:, 0::2] = np.sin(pos * div[: (F_v + 1) // 2])
    out[:, 1::2] = np.cos(pos * div[: F_v // 2])
    return out


class EHRshotHeteroDataset(Dataset):
    """Per-patient heterogeneous EHR-graph windows.

    Args:
      data_path: pickle from scripts/dblp/preprocess_ehrshot.py.
      split: "train" | "val" | "test", per the preprocessed split map.
      history_size / num_preds: window lengths (history is visible, num_preds
        is JEPA-masked).
      max_visits_per_patient: truncate each patient to their LAST K visits.
        0 disables truncation.
      max_windows_per_patient: cap windows per patient (training: uniform
        subsample; val/test: most recent).
      visit_feat_dim: dimensionality of the sinusoidal position embedding on
        visit nodes.
      seed: RNG seed for the per-patient train subsampling.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        history_size: int = 3,
        num_preds: int = 1,
        max_visits_per_patient: int = 60,
        max_windows_per_patient: int = 16,
        visit_feat_dim: int = 16,
        seed: int = 42,
    ):
        _require_pyg()
        assert split in ("train", "val", "test")
        path = Path(data_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"EHRshot pickle not found at {path}. Run preprocess_ehrshot.py first.")
        with open(path, "rb") as f:
            payload = pickle.load(f)

        self.split = split
        self.history_size = history_size
        self.num_preds = num_preds
        self.window = history_size + num_preds
        self.visit_feat_dim = visit_feat_dim
        self.code_vocab: list = payload["code_vocab"]
        self.num_nodes = len(self.code_vocab)
        self._code_feats = _concept_static_features(
            payload["code_vocab_type"], payload["code_freq"], self.num_nodes,
        )
        # Precompute one sinusoidal table for all windows (T is fixed).
        self._visit_pos = _visit_position_features(self.window, visit_feat_dim)

        pids = payload["patient_ids"]
        splits = payload["patient_split"]
        visits_by_pid: dict = payload["patient_visits"]
        keep = [pid for pid, s in zip(pids, splits) if s == split]
        if not keep:
            raise ValueError(f"No patients in split={split!r}.")

        self.patient_ids = keep
        # Store per-patient visit codes as plain Python lists and expand to
        # edge tensors in __getitem__ (cheaper than dense T×N masks).
        self._patient_visits: List[List[List[int]]] = []
        self.starts: list = []
        rng = np.random.default_rng(seed)

        for p_idx, pid in enumerate(keep):
            seq = visits_by_pid[pid]
            if max_visits_per_patient > 0 and len(seq) > max_visits_per_patient:
                seq = seq[-max_visits_per_patient:]
            codes_per_visit = [list(c) for _, c in seq]
            T_p = len(codes_per_visit)
            if T_p < self.window:
                self._patient_visits.append([])
                continue
            self._patient_visits.append(codes_per_visit)
            n_windows = T_p - self.window + 1
            window_starts = np.arange(n_windows)
            if (max_windows_per_patient > 0
                    and len(window_starts) > max_windows_per_patient):
                if split == "train":
                    sel = np.linspace(0, len(window_starts) - 1,
                                       max_windows_per_patient, dtype=np.int64)
                else:
                    sel = np.arange(len(window_starts) - max_windows_per_patient,
                                    len(window_starts))
                window_starts = window_starts[sel]
            self.starts.extend([(p_idx, int(s)) for s in window_starts])

        self.starts = np.array(self.starts, dtype=np.int64) \
            if self.starts else np.zeros((0, 2), dtype=np.int64)
        if self.starts.size == 0:
            raise ValueError(f"Split {split!r} has no valid windows.")

        # Feature_dim semantics that downstream code expects.
        self.feature_dim_total = self._code_feats.shape[1] + visit_feat_dim
        self.year_offset = 0
        self.node_labels = None
        print(f"[ehrshot-hetero] split={split}  patients={len(keep)}  "
              f"windows={len(self.starts)}  N_codes={self.num_nodes}  "
              f"F_code={self._code_feats.shape[1]}  F_visit={visit_feat_dim}",
              flush=True)

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, idx: int) -> Any:
        p_idx, t_start = self.starts[idx]
        codes_per_visit = self._patient_visits[int(p_idx)]
        t_start = int(t_start)
        T = self.window
        H = self.history_size

        # Build (2, E_full) edge tensor for all T visits. v in [0, T), c in [0, N).
        # Visits are indexed locally in the window, so the encoder does not
        # see patient-specific absolute time.
        src_v_full, dst_c_full = [], []
        for t in range(T):
            for c in codes_per_visit[t_start + t]:
                src_v_full.append(t); dst_c_full.append(int(c))
        # Visible edges = those whose visit index < H.
        history_mask = np.array([v < H for v in src_v_full], dtype=bool)
        full_edge = np.array([src_v_full, dst_c_full], dtype=np.int64)         # (2, E_full)
        hist_edge = full_edge[:, history_mask]                                  # (2, E_hist)

        # HeteroData with visible (history-only) edges. The future edges are
        # carried by the dense (T, N) target_mask below.
        data = HeteroData()
        data["visit"].x = torch.from_numpy(self._visit_pos)                     # (T, F_v)
        data["code"].x = torch.from_numpy(self._code_feats)                     # (N, F_c)
        data["visit", "records", "code"].edge_index = torch.from_numpy(hist_edge)
        # Reverse edge for bidirectional message passing in HeteroConv.
        data["code", "rev_records", "visit"].edge_index = torch.from_numpy(
            hist_edge[[1, 0]].copy())
        # Dense (T, N) ground truth of which codes are at which visit (all T
        # visits, including the masked future ones). PyG batches it to (B*T, N).
        target_mask = np.zeros((T, self.num_nodes), dtype=bool)
        for t in range(T):
            target_mask[t, codes_per_visit[t_start + t]] = True
        data["visit"].target_mask = torch.from_numpy(target_mask)               # (T, N)
        # Scalar per-item bookkeeping (concatenated into (B,) tensors).
        data.history_size = torch.tensor([H], dtype=torch.long)
        data.window = torch.tensor([T], dtype=torch.long)
        return data

    @property
    def feature_dim(self) -> int:
        return self.feature_dim_total
