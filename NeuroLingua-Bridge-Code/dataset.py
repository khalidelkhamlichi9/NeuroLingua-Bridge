"""
NeuroLingua-Bridge — dataset.py
ABIDE-I dataset: download via Nilearn, preprocessing, fMRI-text descriptors,
inter-site split and PyTorch DataLoaders.

Preprocessing follows Wei et al. arXiv:2511.21760v3 (2026):
  - CPAC pipeline, CC200 atlas (Craddock 200, 200 ROIs, native)
  - Resample to T=160 timepoints
  - 3 FC matrices: Correlation (Fisher-Z), Tangent, Partial Correlation
  - Robust IQR z-score per site
  - 4 text descriptor types: FC, FG, ICA, Graph
  - Inter-site split: large sites (top 70 % subjects) → Train | small → Test
"""

import os, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.linalg import eigh as sp_eigh
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import networkx as nx

try:
    from nilearn.datasets import fetch_abide_pcp
    from nilearn.connectome import ConnectivityMeasure
    NILEARN_OK = True
except ImportError:
    NILEARN_OK = False

SEED        = 42
N_ROIS      = 200
T_TARGET    = 160
MIN_TP      = 50
TRAIN_RATIO = 0.70

NET_NAMES = ["Visual", "SomMot", "DorsAttn", "SalVentAttn",
             "Limbic", "Cont", "Default", "Subcort"]
# CC200 has ~200 ROIs; partition into 8 contiguous functional blocks of 25.
_BLK = N_ROIS // 8
NET_ROIS  = [list(range(i * _BLK, (i + 1) * _BLK)) for i in range(7)] + \
            [list(range(7 * _BLK, N_ROIS))]

# ICA networks: 7 overlapping blocks across the 200 ROIs.
ICA_ROIS  = [list(range(i * _BLK, (i + 1) * _BLK)) for i in range(7)]


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resample(ts, T=T_TARGET):
    if ts.shape[0] == T: return ts
    f = interp1d(np.linspace(0, 1, ts.shape[0]), ts, axis=0, kind="linear")
    return f(np.linspace(0, 1, T)).astype(np.float32)

def _fit_rois(ts, N=N_ROIS):
    """CC200 is natively ~200 ROIs. Pad or truncate to exactly N without
    fabricating a cross-region interpolation (the AAL->450 hack is removed)."""
    T, n = ts.shape
    if n == N:
        return ts.astype(np.float32)
    if n > N:
        return ts[:, :N].astype(np.float32)
    out = np.zeros((T, N), np.float32)
    out[:, :n] = ts
    return out

def _robust_zscore(ts):
    q75, q25 = np.percentile(ts, 75, axis=0), np.percentile(ts, 25, axis=0)
    return (ts - np.median(ts, axis=0)) / (q75 - q25 + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Text descriptor builders  (Table 1, Wei et al. 2026)
# ─────────────────────────────────────────────────────────────────────────────

def _fc_descriptor(fc):
    pv = {f"{n1}-{n2}": float(fc[np.ix_(r1, r2)].mean())
          for i, (n1, r1) in enumerate(zip(NET_NAMES, NET_ROIS))
          for j, (n2, r2) in enumerate(zip(NET_NAMES, NET_ROIS)) if i < j}
    top3 = sorted(pv, key=pv.get, reverse=True)[:3]
    bot3 = sorted(pv, key=pv.get)[:3]
    return (f"FC top=[{','.join(f'{k}({pv[k]:+.2f})' for k in top3)}] "
            f"bot=[{','.join(f'{k}({pv[k]:+.2f})' for k in bot3)}].")

def _fg_descriptor(fc):
    fc_pos = np.maximum(fc, 0.); np.fill_diagonal(fc_pos, 0.)
    Ln = np.eye(fc.shape[0]) - fc_pos / (fc_pos.sum(1, keepdims=True) + 1e-8)
    try:
        _, V = sp_eigh(Ln, subset_by_index=[1, 4])
        g = [float(V[:, k].max() - V[:, k].min()) for k in range(min(3, V.shape[1]))]
        while len(g) < 3: g.append(0.)
    except Exception:
        g = [0., 0., 0.]
    return f"FG g1={g[0]:.3f} g2={g[1]:.3f} g3={g[2]:.3f}."

def _ica_descriptor(ts):
    sigs = [ts[:, [c for c in r if c < ts.shape[1]]].mean(1)
            if any(c < ts.shape[1] for c in r) else np.zeros(ts.shape[0])
            for r in ICA_ROIS]
    amp = [float(np.abs(s).mean()) for s in sigs]
    var = [float(s.std()) for s in sigs]
    fnc = float(np.corrcoef(np.array(sigs))[np.triu_indices(len(sigs), k=1)].mean())
    return f"ICA amp={np.mean(amp):.3f} var={np.mean(var):.3f} FNC={fnc:.3f}."

def _graph_descriptor(fc):
    np.fill_diagonal(fc, 0.)
    cut = np.percentile(np.abs(fc[np.triu_indices(fc.shape[0], k=1)]), 90)
    A   = np.where(np.abs(fc) >= cut, np.abs(fc), 0.)
    G   = nx.from_numpy_array(A)
    try:
        eff   = nx.global_efficiency(G)
        clust = nx.average_clustering(G, weight="weight")
        parts = nx.community.louvain_communities(G, weight="weight", seed=SEED)
        mod   = nx.community.modularity(G, parts, weight="weight")
    except Exception:
        eff = clust = mod = 0.
    return f"Graph mod={mod:.3f} eff={eff:.3f} clust={clust:.3f}."

def build_descriptor(ts, fc):
    return " ".join([_fc_descriptor(fc), _fg_descriptor(fc),
                     _ica_descriptor(ts), _graph_descriptor(fc)])


# ─────────────────────────────────────────────────────────────────────────────
# ABIDE-I download + full preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_abide(data_dir: str = "./abide_data"):
    """
    Download ABIDE-I (via Nilearn) and return fully preprocessed arrays.

    Returns
    -------
    ts_all      : (N, T, ROI) float32
    fc_corr_all : (N, ROI, ROI) float32
    fc_tan_all  : (N, ROI, ROI) float32
    fc_par_all  : (N, ROI, ROI) float32
    lbl_all     : (N,) int64  — 1=ASD, 0=Control
    sex_all     : (N,) int64
    site_list   : list[str]
    all_descs   : list[str]
    """
    assert NILEARN_OK, "pip install nilearn"
    Path(data_dir).mkdir(exist_ok=True)
    print("Downloading ABIDE-I (pipeline=cpac, rois_cc200, 200 ROIs)...")
    abide    = fetch_abide_pcp(data_dir=data_dir, pipeline="cpac",
                               band_pass_filtering=True,
                               global_signal_regression=False,
                               derivatives=["rois_cc200"],
                               quality_checked=True, verbose=0)
    pheno    = pd.DataFrame(abide.phenotypic)
    dx_arr   = pd.to_numeric(pheno["DX_GROUP"],  errors="coerce").fillna(0).values
    sex_raw  = pd.to_numeric(pheno["SEX"], errors="coerce").fillna(2).astype(int).values
    site_raw = pheno.get("SITE_ID", pd.Series(["UNK"] * len(pheno))).values

    valid_idx = [i for i, ts in enumerate(abide.rois_cc200)
                 if np.array(ts).ndim == 2 and np.array(ts).shape[0] > 10]
    roi_all  = [abide.rois_cc200[i] for i in valid_idx]
    sex_v    = sex_raw[valid_idx]
    site_v   = site_raw[valid_idx]

    conn_c = ConnectivityMeasure(kind="correlation",         vectorize=False)
    conn_p = ConnectivityMeasure(kind="partial correlation", vectorize=False)
    buf    = {"ts": [], "fc_c": [], "fc_p": [], "lbl": [], "sex": [], "site": []}

    for i, roi_data in enumerate(tqdm(roi_all, desc="Preprocessing")):
        try:
            ts = np.array(roi_data, dtype=np.float32)
            if ts.shape[0] < ts.shape[1]: ts = ts.T
            if ts.shape[0] < MIN_TP: continue
            ts = _resample(ts)
            ts = _fit_rois(ts)
            ts = _robust_zscore(ts).astype(np.float32)

            fc_c = conn_c.fit_transform([ts])[0]
            np.fill_diagonal(fc_c, 0.)
            fc_c = np.arctanh(np.clip(fc_c, -0.999, 0.999)).astype(np.float32)

            try:
                fc_p = conn_p.fit_transform([ts])[0]
                np.fill_diagonal(fc_p, 0.)
                fc_p = fc_p.astype(np.float32)
            except Exception:
                fc_p = fc_c.copy()

            lbl = 1 if int(dx_arr[valid_idx[i]]) == 1 else 0
            buf["ts"].append(ts); buf["fc_c"].append(fc_c)
            buf["fc_p"].append(fc_p); buf["lbl"].append(lbl)
            buf["sex"].append(int(sex_v[i])); buf["site"].append(site_v[i])
        except Exception:
            continue

    # Tangent FC (batch)
    conn_t = ConnectivityMeasure(kind="tangent", vectorize=False)
    try:
        fc_t_list = conn_t.fit_transform(buf["ts"])
        fc_t_list = [f.astype(np.float32) for f in fc_t_list]
        for f in fc_t_list: np.fill_diagonal(f, 0.)
    except Exception:
        fc_t_list = [f.copy() for f in buf["fc_c"]]

    ts_all      = np.array(buf["ts"],   dtype=np.float32)
    fc_corr_all = np.array(buf["fc_c"], dtype=np.float32)
    fc_tan_all  = np.array(fc_t_list,   dtype=np.float32)
    fc_par_all  = np.array(buf["fc_p"], dtype=np.float32)
    lbl_all     = np.array(buf["lbl"],  dtype=np.int64)
    sex_all     = np.array(buf["sex"],  dtype=np.int64)
    site_list   = buf["site"]

    # Site-wise variance normalisation
    for s in np.unique(site_list):
        mask = np.array(site_list) == s
        if mask.sum() >= 2:
            ts_all[mask] /= (ts_all[mask].std(axis=0, keepdims=True) + 1e-8)

    print("Building fMRI-text descriptors...")
    all_descs = [build_descriptor(ts_all[i], fc_corr_all[i])
                 for i in tqdm(range(len(ts_all)), desc="Descriptors")]

    print(f"Done: {len(ts_all)} subjects  ASD={lbl_all.sum()}  "
          f"CTL={(lbl_all == 0).sum()}  sites={len(np.unique(site_list))}")
    return ts_all, fc_corr_all, fc_tan_all, fc_par_all, lbl_all, sex_all, site_list, all_descs


# ─────────────────────────────────────────────────────────────────────────────
# Inter-site split  (Wei et al. 2026 protocol)
# ─────────────────────────────────────────────────────────────────────────────

def inter_site_split(lbl_all, site_list, train_ratio: float = TRAIN_RATIO):
    """
    Adaptive inter-site split.
    Large sites (top `train_ratio` of subjects) → Train + Val.
    Small sites → Test  (unseen domain, domain-shift evaluation).
    """
    site_arr    = np.array(site_list)
    site_counts = pd.Series(site_list).value_counts().sort_values(ascending=False)
    cum_pct     = site_counts.cumsum() / len(site_list)

    big_sites   = site_counts[cum_pct <= train_ratio].index.tolist()
    small_sites = site_counts[cum_pct >  train_ratio].index.tolist()

    if len(small_sites) < 2:
        n = max(2, len(site_counts) // 4)
        small_sites = site_counts.index[-n:].tolist()
        big_sites   = site_counts.index[:-n].tolist()

    train_full = np.where(np.isin(site_arr, big_sites))[0]
    test_idx   = np.where(np.isin(site_arr, small_sites))[0]

    cls, cnt = np.unique(lbl_all[train_full], return_counts=True)
    strat = len(cls) >= 2 and cnt.min() >= 2 and int(len(train_full) * 0.15) >= 2
    if strat:
        train_idx, val_idx = train_test_split(
            train_full, test_size=0.15,
            stratify=lbl_all[train_full], random_state=SEED)
    else:
        sz = max(2, int(len(train_full) * 0.15))
        train_idx, val_idx = train_full[sz:], train_full[:sz]

    print(f"Split — Train:{len(train_idx)}  Val:{len(val_idx)}  Test:{len(test_idx)}")
    print(f"  Train sites ({len(big_sites)}): {big_sites}")
    print(f"  Test  sites ({len(small_sites)}): {small_sites}")
    return train_idx, val_idx, test_idx, big_sites, small_sites


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset + DataLoaders
# ─────────────────────────────────────────────────────────────────────────────

class ABIDEDataset(Dataset):
    def __init__(self, ts, fc_corr, fc_tan, fc_par, labels, sex,
                 descs, tokenizer=None, max_len: int = 256, augment: bool = False):
        self.ts, self.fc_corr = ts.astype(np.float32), fc_corr.astype(np.float32)
        self.fc_tan, self.fc_par = fc_tan.astype(np.float32), fc_par.astype(np.float32)
        self.labels  = np.array(labels, dtype=np.int64)
        self.sex     = np.clip(np.array(sex) - 1, 0, 1).astype(np.int64)
        self.descs   = descs
        self.augment = augment
        if tokenizer is not None:
            self.tok_ids = np.array([
                tokenizer(d, max_length=max_len, truncation=True,
                          padding="max_length", return_tensors="np")["input_ids"][0]
                for d in descs], dtype=np.int64)
        else:
            self.tok_ids = np.zeros((len(labels), max_len), dtype=np.int64)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        ts = torch.from_numpy(self.ts[i]).clone()
        if self.augment:
            ts = ts + torch.randn_like(ts) * 0.02
            if torch.rand(1) > 0.5: ts = ts.flip(0)
        return {"ts":      ts,
                "fc_corr": torch.from_numpy(self.fc_corr[i]),
                "fc_tan":  torch.from_numpy(self.fc_tan[i]),
                "fc_par":  torch.from_numpy(self.fc_par[i]),
                "label":   torch.tensor(self.labels[i]),
                "sex":     torch.tensor(self.sex[i]),
                "tok_ids": torch.from_numpy(self.tok_ids[i])}


def make_loaders(ts_all, fc_corr_all, fc_tan_all, fc_par_all,
                 lbl_all, sex_all, all_descs,
                 train_idx, val_idx, test_idx,
                 tokenizer=None, batch_size: int = 4, max_len: int = 256):
    def _mk(idx, aug, shuf):
        ds = ABIDEDataset(ts_all[idx], fc_corr_all[idx], fc_tan_all[idx], fc_par_all[idx],
                          lbl_all[idx], sex_all[idx], [all_descs[j] for j in idx],
                          tokenizer=tokenizer, max_len=max_len, augment=aug)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuf,
                          drop_last=False, pin_memory=torch.cuda.is_available())
    return _mk(train_idx, True, True), _mk(val_idx, False, False), _mk(test_idx, False, False)
