"""Dataset dynamics audit: how much is there for a latent FORECASTER to do?

For each dataset the script answers, from the (T, N) dynamic label matrix and
the (T, N, N) snapshots alone (no trained model):

  1. label dynamics    persistence floor (overall / test), switch rate, how
                       many nodes ever switch and how many switch again
                       (recurring state vs one-way drift), persistence at
                       lags 1..L, class-transition matrix of the switches
  2. label-history     a logistic probe on a node's LAST k LABELS only
     control           (k = 1, 2, 3) -> label at t. The ceiling a graph-free
                       autoregressive model reaches; its accuracy on the
                       switch subset says whether switches are predictable
                       from label history at all
  3. graph lead        the same probe fed the node's neighbourhood at t-1
                       (neighbour-label histogram at t-1 and t-2, degree and
                       its change, share of neighbours that just switched)
                       plus its own label at t-1. The gain over (2) on the
                       switch subset is the STRUCTURAL lead signal: what a
                       graph encoder can add to label persistence. Optionally
                       (--feats) the release node features at t-1 as well
  4. structure         consecutive Jaccard, edges new / gone, share of
                       all-time edges per snapshot, Jaccard at lags 1..L,
                       degree autocorrelation
  5. coverage          labelled cells per snapshot (first / last / test),
                       active nodes, share of all switch events in the test
                       snapshots

On bipartite sets (JODIE) only users carry labels, so the neighbour-label
histogram in (3) is empty by construction and the graph probe reduces to
degree dynamics; the homophily line prints "--" there.

Every feature in (3) is observed at t-1 or earlier: nothing from the target
snapshot leaks in (no "oracle" column, per the repo's table rule).

Usage (repo root, PYTHONPATH=.):
  python scripts/analysis/dataset_dynamics.py --datasets chase_music
  python scripts/analysis/dataset_dynamics.py \
      --datasets mooc reddit wikipedia chase_music chase_gossipcop chase_dblp_s \
                 bts_flights nyc_taxi --out outputs/dataset_dynamics.json

Dataset knobs follow the launchers' defaults (scripts/dblp/_dataset_config.sh):
JODIE 30 bins with STATE_MODE=state; CHASE full graphs; BTS 2019 daily; NYC
taxi 2019-01..03 hourly. Override any factory flag after the dataset list
(e.g. --jodie_num_bins 60, --chase_top_k 8000). DBLP-S and the full JODIE
sets need a cluster node: the dense snapshots alone are several GB.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.custom_codes.dataset_factory import add_dataset_cli_flags, build_dynamic_dataset  # noqa: E402

ALIASES = {"mooc": "jodie_mooc", "reddit": "jodie_reddit",
           "wikipedia": "jodie_wikipedia", "patent": "cldg_patent"}
PROBE_C = 1.0
SPLITS_DEFAULT = (0.7, 0.1, 0.2)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def build(name: str, extra: list[str], all_cells: bool = False):
    """Build the dataset through the factory with the launchers' defaults and
    return (adj, active, labels, feats, n_tr, n_va, meta)."""
    p = argparse.ArgumentParser()
    add_dataset_cli_flags(p)
    base = ["--data_source", name]
    if name.startswith("jodie_"):
        base += ["--jodie_data_path", f"dataset/jodie/{name.split('_', 1)[1]}.csv",
                 "--with_state_labels", "--jodie_state_label_mode", "state"]
    args, unknown = p.parse_known_args(base + extra)
    if unknown:
        raise SystemExit(f"unknown dataset flags: {unknown}")
    for k, v in dict(seed=42, with_edge_features=False, split_mode="chronological",
                     coarse_window=1, history_size=1, num_preds=1).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    ds = build_dynamic_dataset(args, "train")
    inner = getattr(ds, "_ds", ds)
    adj = np.asarray(inner.adj)
    active = np.asarray(inner.active).astype(bool)
    labels = getattr(ds, "node_labels_dyn", None)
    if labels is None:
        raise SystemExit(f"{name}: no dynamic label matrix (ds.node_labels_dyn); "
                         "nothing to audit")
    labels = np.asarray(labels).astype(np.int64)
    # Score only nodes ACTIVE at the labelled snapshot, as the pipeline's probe
    # does (run_block_probe collects active rows at the target step). The
    # JODIE state matrix labels every user at every bin, so without this the
    # inactive majority pads persistence (MOOC: 7047 labelled vs ~1000 active).
    if not all_cells:
        labels = np.where(active, labels, -1)
    T = adj.shape[0]
    if hasattr(ds, "split_counts"):
        n_tr, n_va, _ = ds.split_counts
    else:
        s = tuple(getattr(ds, "SPLITS", SPLITS_DEFAULT))
        n_tr = int(T * s[0]); n_va = int(T * s[1])
    feats = np.asarray(inner.feats) if getattr(inner, "feats", None) is not None else None
    meta = dict(state_label_mode=str(getattr(ds, "state_label_mode", "")),
                feat_dim=int(feats.shape[-1]) if feats is not None else 0)
    return adj, active, labels, feats, n_tr, n_va, meta


# --------------------------------------------------------------------------
# 1. label dynamics
# --------------------------------------------------------------------------
def label_dynamics(labels, n_tr, n_va, max_lag):
    T, N = labels.shape
    v = labels >= 0
    C = int(labels.max()) + 1
    out = {"T": T, "N": N, "classes": C,
           "labelled_share": float(v.mean()),
           "train_class_freq": (np.bincount(labels[:n_tr][v[:n_tr]], minlength=C)
                                / max(v[:n_tr].sum(), 1)).round(4).tolist()}
    both = v[1:] & v[:-1]
    same = (labels[1:] == labels[:-1]) & both
    ch = (labels[1:] != labels[:-1]) & both
    te0 = n_tr + n_va - 1                        # transitions INTO test snapshots
    out["persistence_acc"] = float(same.sum() / max(both.sum(), 1))
    out["persistence_acc_test"] = float(same[te0:].sum() / max(both[te0:].sum(), 1))
    out["switch_rate"] = float(ch.sum() / max(both.sum(), 1))
    out["switch_rate_test"] = float(ch[te0:].sum() / max(both[te0:].sum(), 1))
    n_ch = ch.sum(0)
    ever = v.any(0)
    out["nodes_1plus_changes"] = float((n_ch >= 1)[ever].mean())
    out["nodes_2plus_changes"] = float((n_ch >= 2)[ever].mean())
    out["recurring_share"] = float((n_ch >= 2).sum() / max((n_ch >= 1).sum(), 1))
    out["mean_changes_per_switching_node"] = float(n_ch[n_ch >= 1].mean()) if (n_ch >= 1).any() else 0.0
    out["switch_events"] = int(ch.sum())
    out["switch_events_in_test_share"] = float(ch[te0:].sum() / max(ch.sum(), 1))
    # persistence at lag k: label(t) == label(t-k) over cells valid at both
    lags = {}
    for k in range(1, min(max_lag, T - 1) + 1):
        b = v[k:] & v[:-k]
        lags[k] = float(((labels[k:] == labels[:-k]) & b).sum() / max(b.sum(), 1))
    out["persistence_at_lag"] = lags
    # transition matrix among switches, row = from, col = to
    if ch.any():
        fr = labels[:-1][ch]; to = labels[1:][ch]
        M = np.zeros((C, C), dtype=np.int64)
        np.add.at(M, (fr, to), 1)
        out["switch_transitions"] = M.tolist()
        if C == 2:
            out["onset_share_of_switches"] = float(M[0, 1] / max(M.sum(), 1))
    return out


# --------------------------------------------------------------------------
# 2 + 3. probes
# --------------------------------------------------------------------------
def _onehot(x, C):
    """One-hot with an extra 'unknown' column for -1."""
    o = np.zeros((len(x), C + 1), dtype=np.float32)
    o[np.arange(len(x)), np.where(x >= 0, x, C)] = 1.0
    return o


def _neighbour_hist(adj_t, labels_t, C):
    """(N, C+1) counts of neighbour labels at one snapshot (col C = unlabelled)."""
    import scipy.sparse as sp
    A = sp.csr_matrix(adj_t)
    return np.asarray(A @ _onehot(labels_t, C)), np.asarray(A.sum(1)).ravel()


def _fit_eval(X_tr, y_tr, X_te, y_te, switch_te, C, prev_te=None):
    """Logistic probe. Besides accuracy / macro-F1 / switch- and stay-subset
    accuracy, reports the SWITCH ALARM AUROC: P(label at t != label at t-1)
    = 1 - P(prev label) under the probe, scored against the switch indicator.
    It isolates 'is a change coming' from 'which class comes next'."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=2000, C=PROBE_C)
    clf.fit(sc.transform(X_tr), y_tr)
    proba = clf.predict_proba(sc.transform(X_te))
    pred = clf.classes_[proba.argmax(1)]
    r = {"acc": float((pred == y_te).mean()),
         "macro_f1": float(f1_score(y_te, pred, average="macro")),
         "switch_acc": float((pred[switch_te] == y_te[switch_te]).mean()) if switch_te.any() else float("nan"),
         "stay_acc": float((pred[~switch_te] == y_te[~switch_te]).mean()) if (~switch_te).any() else float("nan"),
         "n_test": int(len(y_te)), "n_switch_test": int(switch_te.sum())}
    if C == 2 and len(np.unique(y_te)) == 2:
        r["auroc"] = float(roc_auc_score(y_te, proba[:, list(clf.classes_).index(1)]))
    if prev_te is not None and switch_te.any() and (~switch_te).any():
        col = {c: j for j, c in enumerate(clf.classes_)}
        p_prev = np.array([proba[i, col[c]] if c in col else 0.0 for i, c in enumerate(prev_te)])
        ok = prev_te >= 0
        r["alarm_auroc"] = float(roc_auc_score(switch_te[ok], 1.0 - p_prev[ok]))
    return r


def probes(adj, active, labels, feats, n_tr, n_va, k_max, with_feats, max_rows, seed, feat_pca=64):
    """Rows = labelled (t, node) cells with t >= k_max (so every row has the
    same history length). Train rows: targets in the train snapshots; test
    rows: targets in the test snapshots. Returns {model: metrics}."""
    T, N = labels.shape
    C = int(labels.max()) + 1
    rng = np.random.default_rng(seed)
    t0 = max(k_max, 2)
    rows_t, rows_i = np.nonzero(labels[t0:] >= 0)
    rows_t = rows_t + t0
    train = rows_t < n_tr
    test = rows_t >= n_tr + n_va
    if train.sum() > max_rows:
        keep = np.zeros_like(train); keep[rng.choice(np.nonzero(train)[0], max_rows, replace=False)] = True
        train = keep
    y = labels[rows_t, rows_i]
    prev = labels[rows_t - 1, rows_i]
    switch = (prev >= 0) & (prev != y)
    # label-history blocks: last k labels one-hot (with 'unknown')
    hist = [_onehot(labels[rows_t - k, rows_i], C) for k in range(1, k_max + 1)]
    out = {}
    maj = int(np.bincount(y[train], minlength=C).argmax())
    # copy-last, falling back to the train-majority class where no previous
    # label exists (so every model is scored on the SAME test rows)
    pers = np.where(prev >= 0, prev, maj)
    out["persistence"] = {
        "acc": float((pers[test] == y[test]).mean()),
        "switch_acc": 0.0, "stay_acc": 1.0,
        "n_test": int(test.sum()), "n_switch_test": int(switch[test].sum())}
    out["majority_class"] = {
        "acc": float((y[test] == maj).mean()),
        "switch_acc": float((y[test][switch[test]] == maj).mean()) if switch[test].any() else float("nan"),
        "stay_acc": float((y[test][~switch[test]] == maj).mean()),
        "n_test": int(test.sum()), "n_switch_test": int(switch[test].sum())}
    for k in range(1, k_max + 1):
        X = np.concatenate(hist[:k], axis=1)
        out[f"labels_last{k}"] = _fit_eval(X[train], y[train], X[test], y[test], switch[test], C, prev[test])
    # graph features at t-1 / t-2 (nothing from t)
    H1 = np.zeros((T, N, C + 1), np.float32); D = np.zeros((T, N), np.float32)
    for t in range(T):
        H1[t], D[t] = _neighbour_hist(adj[t], labels[t], C)
    def nb(t_idx):
        h = H1[t_idx, rows_i]; d = D[t_idx, rows_i]
        return h / np.maximum(d[:, None], 1.0), np.log1p(d)
    h1, ld1 = nb(rows_t - 1); h2, ld2 = nb(rows_t - 2)
    # share of neighbours (at t-1) whose label changed between t-2 and t-1
    chg = ((labels[1:] != labels[:-1]) & (labels[1:] >= 0) & (labels[:-1] >= 0)).astype(np.float32)
    nb_chg = np.zeros((T, N), np.float32)
    import scipy.sparse as sp
    for t in range(1, T):
        A = sp.csr_matrix(adj[t])
        nb_chg[t] = np.asarray(A @ chg[t - 1]).ravel() / np.maximum(D[t], 1.0)
    Xg = np.concatenate([hist[0], h1, h2, h1 - h2, ld1[:, None], ld2[:, None],
                         (ld1 - ld2)[:, None], nb_chg[rows_t - 1, rows_i][:, None]], axis=1)
    out["graph_t-1"] = _fit_eval(Xg[train], y[train], Xg[test], y[test], switch[test], C, prev[test])
    Xg2 = np.concatenate([Xg] + hist[1:min(k_max, 2)], axis=1)
    out["graph_t-1+labels_last2"] = _fit_eval(Xg2[train], y[train], Xg2[test], y[test], switch[test], C, prev[test])
    # homophily lead: do switchers move to their t-1 neighbourhood's majority label?
    sw = switch & test
    if sw.any():
        maj = h1[sw][:, :C].argmax(1)
        has_nb = h1[sw][:, :C].sum(1) > 0
        dest_freq = np.bincount(y[sw], minlength=C) / max(sw.sum(), 1)
        out["homophily_lead"] = {
            "switchers_to_neighbour_majority": float((maj == y[sw])[has_nb].mean()) if has_nb.any() else float("nan"),
            "chance_(dest_class_freq_of_majority)": float(dest_freq[maj][has_nb].mean()) if has_nb.any() else float("nan"),
            "switchers_with_labelled_neighbours": float(has_nb.mean())}
    if with_feats and feats is not None:
        # PCA (fit on train rows) keeps the probe from memorising a few
        # thousand rows with ~800 raw dims; the label one-hot is appended raw.
        from sklearn.decomposition import PCA
        F = feats[rows_t - 1, rows_i].astype(np.float32)
        F = np.nan_to_num(F)
        n_pc = F.shape[1] if feat_pca <= 0 else min(feat_pca, F.shape[1], max(2, train.sum() - 1))
        if n_pc < F.shape[1]:
            F = PCA(n_components=n_pc, random_state=seed).fit(F[train]).transform(F)
        Xf = np.concatenate([hist[0], F], axis=1)
        out["feats_t-1"] = _fit_eval(Xf[train], y[train], Xf[test], y[test], switch[test], C, prev[test])
        Xfg = np.concatenate([Xg, F], axis=1)
        out["graph+feats_t-1"] = _fit_eval(Xfg[train], y[train], Xfg[test], y[test], switch[test], C, prev[test])
    out["_rows"] = {"train": int(train.sum()), "test": int(test.sum()), "C": C}
    return out


# --------------------------------------------------------------------------
# 4 + 5. structure and coverage
# --------------------------------------------------------------------------
def structure(adj, active, labels, n_tr, n_va, max_lag):
    T, N = active.shape
    iu = np.triu_indices(N, k=1)
    E = [np.flatnonzero(adj[t][iu]) for t in range(T)]      # upper-triangle edge ids
    sizes = np.array([len(e) for e in E])
    all_edges = np.unique(np.concatenate(E)) if sizes.sum() else np.zeros(0, np.int64)
    jac = {}
    for k in range(1, min(max_lag, T - 1) + 1):
        js = []
        for t in range(k, T):
            a, b = E[t - k], E[t]
            inter = np.intersect1d(a, b, assume_unique=True).size
            uni = a.size + b.size - inter
            js.append(inter / uni if uni else float("nan"))
        jac[k] = float(np.nanmean(js))
    new = []; gone = []
    for t in range(1, T):
        inter = np.intersect1d(E[t - 1], E[t], assume_unique=True).size
        new.append(1 - inter / max(E[t].size, 1)); gone.append(1 - inter / max(E[t - 1].size, 1))
    deg = adj.sum(-1).astype(np.float32)                    # (T, N)
    ac = []
    for t in range(1, T):
        m = active[t] & active[t - 1]
        if m.sum() > 2 and deg[t][m].std() > 0 and deg[t - 1][m].std() > 0:
            ac.append(np.corrcoef(deg[t][m], deg[t - 1][m])[0, 1])
    v = labels >= 0
    te0 = n_tr + n_va
    return {"edges_per_snapshot_mean": float(sizes.mean()),
            "all_time_edges": int(all_edges.size),
            "share_of_all_time_edges_per_snapshot": float(sizes.mean() / max(all_edges.size, 1)),
            "jaccard_at_lag": jac,
            "edges_new_share": float(np.mean(new)), "edges_gone_share": float(np.mean(gone)),
            "degree_autocorr_lag1": float(np.mean(ac)) if ac else float("nan"),
            "active_per_snapshot": {"first": int(active[0].sum()), "last": int(active[-1].sum()),
                                    "min": int(active.sum(1).min()), "max": int(active.sum(1).max())},
            "labelled_per_snapshot": {"first": int(v[0].sum()), "last": int(v[-1].sum()),
                                      "train_mean": float(v[:n_tr].sum(1).mean()),
                                      "test_mean": float(v[te0:].sum(1).mean())}}


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def fmt(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def print_block(name, L, P, S, n_tr, n_va, meta, secs):
    T = L["T"]
    print(f"\n=== {name}  T={T} ({n_tr}/{n_va}/{T - n_tr - n_va})  N={L['N']}  "
          f"classes={L['classes']}  labels={meta['state_label_mode']}  ({secs:.0f}s)")
    print(f"  labels    labelled={L['labelled_share']:.2f}  train class freq={L['train_class_freq']}")
    print(f"            persistence={L['persistence_acc']:.3f} (test {L['persistence_acc_test']:.3f})  "
          f"switch rate={L['switch_rate']:.3f} (test {L['switch_rate_test']:.3f})")
    print(f"            nodes with 1+ changes={L['nodes_1plus_changes']:.2f}  2+={L['nodes_2plus_changes']:.2f}  "
          f"recurring (2+ among 1+)={L['recurring_share']:.2f}  mean changes/switcher={L['mean_changes_per_switching_node']:.2f}")
    print(f"            switch events={L['switch_events']}  share in test snapshots={L['switch_events_in_test_share']:.2f}")
    print("            persistence at lag k: " + "  ".join(f"{k}:{v:.3f}" for k, v in L["persistence_at_lag"].items()))
    if "switch_transitions" in L and L["classes"] <= 6:
        M = np.array(L["switch_transitions"]); Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
        print("            switch transitions (row=from, col=to, row-normalised):")
        for i in range(L["classes"]):
            print(f"              {i}: " + " ".join(f"{x:.2f}" for x in Mn[i]) + f"   (n={M[i].sum()})")
    print(f"  structure Jaccard lag k: " + "  ".join(f"{k}:{v:.3f}" for k, v in S["jaccard_at_lag"].items()))
    print(f"            edges new={S['edges_new_share']:.3f} gone={S['edges_gone_share']:.3f}  "
          f"share of all-time edges/snapshot={S['share_of_all_time_edges_per_snapshot']:.3f}  "
          f"degree autocorr={fmt(S['degree_autocorr_lag1'])}")
    a, l = S["active_per_snapshot"], S["labelled_per_snapshot"]
    print(f"  coverage  active first/last={a['first']}/{a['last']}  labelled first/last={l['first']}/{l['last']}  "
          f"labelled per snapshot train/test={l['train_mean']:.0f}/{l['test_mean']:.0f}")
    if P:
        print(f"  probes    (test rows={P['_rows']['test']}, switch rows={P['persistence']['n_switch_test']})")
        print(f"            {'model':<26}{'acc':>7}{'macroF1':>9}{'switch':>8}{'stay':>7}{'auroc':>7}{'alarm':>7}")
        for m, r in P.items():
            if m.startswith("_") or m == "homophily_lead":
                continue
            print(f"            {m:<26}{fmt(r.get('acc')):>7}{fmt(r.get('macro_f1')):>9}"
                  f"{fmt(r.get('switch_acc')):>8}{fmt(r.get('stay_acc')):>7}{fmt(r.get('auroc')):>7}"
                  f"{fmt(r.get('alarm_auroc')):>7}")
        h = P.get("homophily_lead")
        if h:
            print(f"            switchers moving to t-1 neighbour-majority label: "
                  f"{fmt(h['switchers_to_neighbour_majority'])} (chance {fmt(h['chance_(dest_class_freq_of_majority)'])}, "
                  f"{h['switchers_with_labelled_neighbours']:.2f} have labelled neighbours)")


def summary_table(results):
    cols = ["dataset", "T", "N", "C", "lab%", "persist(te)", "switch%", "nodes1+%", "recur%",
            "jac1", "kept%", "hist-alarm", "graph-alarm", "feat-alarm", "hist-sw", "graph-sw", "feat-sw"]
    rows = []
    for name, r in results.items():
        L, S, P = r["labels"], r["structure"], r.get("probes") or {}
        lh = P.get("labels_last3") or P.get("labels_last2") or P.get("labels_last1") or {}
        g = P.get("graph_t-1+labels_last2") or P.get("graph_t-1") or {}
        f = P.get("graph+feats_t-1") or {}
        rows.append([name, L["T"], L["N"], L["classes"], f"{100 * L['labelled_share']:.0f}",
                     f"{L['persistence_acc_test']:.3f}", f"{100 * L['switch_rate']:.1f}",
                     f"{100 * L['nodes_1plus_changes']:.0f}", f"{100 * L['recurring_share']:.0f}",
                     f"{S['jaccard_at_lag'].get(1, float('nan')):.3f}",
                     f"{100 * S['share_of_all_time_edges_per_snapshot']:.1f}",
                     fmt(lh.get("alarm_auroc")), fmt(g.get("alarm_auroc")), fmt(f.get("alarm_auroc")),
                     fmt(lh.get("switch_acc")), fmt(g.get("switch_acc")), fmt(f.get("switch_acc"))])
    w = [max(len(str(x)) for x in col) for col in zip(cols, *rows)]
    print("\n" + "  ".join(c.rjust(n) for c, n in zip(cols, w)))
    for r in rows:
        print("  ".join(str(x).rjust(n) for x, n in zip(r, w)))
    print("\n  lab% labelled cells; persist(te) copy-last accuracy on test transitions; switch% of "
          "consecutive labelled pairs; nodes1+% ever switch; recur% of those switch again;\n  jac1 "
          "consecutive-snapshot edge Jaccard; kept% share of all-time edges per snapshot.\n  *-alarm = "
          "AUROC of P(label changes at t) from the probe on the last 3 labels (hist), the t-1 "
          "neighbourhood + last 2 labels (graph),\n  neighbourhood + release features at t-1 (feat, "
          "--feats only); *-sw = the same probes' accuracy on the switch rows. 0.5 alarm = changes "
          "unpredictable from that input.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="factory names or aliases: mooc reddit wikipedia chase_music "
                         "chase_gossipcop chase_dblp_s bts_flights nyc_taxi ...")
    ap.add_argument("--max_lag", type=int, default=6)
    ap.add_argument("--k_labels", type=int, default=3, help="longest label history for the control probe")
    ap.add_argument("--no_probe", action="store_true", help="layers 1, 4, 5 only (no sklearn fits)")
    ap.add_argument("--all_cells", action="store_true",
                    help="keep labels on INACTIVE nodes too (default: only nodes active at that snapshot, "
                         "matching the pipeline probe; matters for the JODIE state matrix)")
    ap.add_argument("--feats", action="store_true", help="also probe the release node features at t-1")
    ap.add_argument("--max_rows", type=int, default=300_000, help="subsample train rows for the probes")
    ap.add_argument("--feat_pca", type=int, default=64, help="PCA dims for the --feats probe (0 = raw)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write every number to this JSON")
    args, extra = ap.parse_known_args()

    results = {}
    for raw in args.datasets:
        name = ALIASES.get(raw, raw)
        t0 = time.time()
        adj, active, labels, feats, n_tr, n_va, meta = build(name, extra, args.all_cells)
        L = label_dynamics(labels, n_tr, n_va, args.max_lag)
        S = structure(adj, active, labels, n_tr, n_va, args.max_lag)
        P = None if args.no_probe else probes(adj, active, labels, feats, n_tr, n_va,
                                              args.k_labels, args.feats, args.max_rows, args.seed, args.feat_pca)
        results[name] = {"labels": L, "structure": S, "probes": P,
                         "splits": [n_tr, n_va, L["T"] - n_tr - n_va], "meta": meta}
        print_block(name, L, P, S, n_tr, n_va, meta, time.time() - t0)
        del adj, active, labels, feats
    if len(results) > 1 or args.out:
        summary_table(results)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1, default=float))
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
