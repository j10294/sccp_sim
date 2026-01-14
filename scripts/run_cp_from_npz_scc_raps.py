#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# Optional sklearn kmeans
try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None


# ----------------------------
# NPZ loader (same spirit as your run_cp_from_npz)
# ----------------------------
def _is_prob_matrix(a: np.ndarray, K: int) -> bool:
    return isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == K and np.isfinite(a).all()

def _is_label_vector(a: np.ndarray) -> bool:
    return isinstance(a, np.ndarray) and a.ndim == 1 and np.issubdtype(a.dtype, np.integer)

def _normalize_rows(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = P.sum(axis=1, keepdims=True)
    s = np.where(s <= eps, 1.0, s)
    Pn = P / s
    Pn = np.clip(Pn, eps, 1.0)
    Pn = Pn / Pn.sum(axis=1, keepdims=True)
    return Pn

def _find_by_name(d: Dict[str, np.ndarray], include: Tuple[str, ...], K: int) -> Optional[np.ndarray]:
    for k in d.keys():
        lk = k.lower()
        if all(s in lk for s in include):
            a = d[k]
            if _is_prob_matrix(a, K):
                return a
    return None

def _find_label_by_name(d: Dict[str, np.ndarray], include: Tuple[str, ...]) -> Optional[np.ndarray]:
    for k in d.keys():
        lk = k.lower()
        if all(s in lk for s in include):
            a = d[k]
            if _is_label_vector(a):
                return a
    return None

def load_npz_splits(path: str, K: int, fallback_split_seed: int = 1) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    raw = np.load(path, allow_pickle=True)
    d = {k: raw[k] for k in raw.files}

    def _pick_first_nonnull(items):
        for x in items:
            if x is not None:
                return x
        return None

    P_sel = _pick_first_nonnull([
        _find_by_name(d, ("p", "sel"), K),
        _find_by_name(d, ("prob", "sel"), K),
        _find_by_name(d, ("probs", "sel"), K),
    ])
    y_sel = _pick_first_nonnull([
        _find_label_by_name(d, ("y", "sel")),
        _find_label_by_name(d, ("label", "sel")),
        _find_label_by_name(d, ("labels", "sel")),
    ])

    P_cal = _pick_first_nonnull([
        _find_by_name(d, ("p", "cal"), K),
        _find_by_name(d, ("prob", "cal"), K),
        _find_by_name(d, ("probs", "cal"), K),
    ])
    y_cal = _pick_first_nonnull([
        _find_label_by_name(d, ("y", "cal")),
        _find_label_by_name(d, ("label", "cal")),
        _find_label_by_name(d, ("labels", "cal")),
    ])

    P_test = _pick_first_nonnull([
        _find_by_name(d, ("p", "test"), K),
        _find_by_name(d, ("prob", "test"), K),
        _find_by_name(d, ("probs", "test"), K),
        _find_by_name(d, ("p", "val"), K),
        _find_by_name(d, ("prob", "val"), K),
        _find_by_name(d, ("probs", "val"), K),
    ])
    y_test = _pick_first_nonnull([
        _find_label_by_name(d, ("y", "test")),
        _find_label_by_name(d, ("label", "test")),
        _find_label_by_name(d, ("labels", "test")),
        _find_label_by_name(d, ("y", "val")),
        _find_label_by_name(d, ("label", "val")),
        _find_label_by_name(d, ("labels", "val")),
    ])

    # heuristic pairing if needed
    prob_keys = [k for k, v in d.items() if _is_prob_matrix(v, K)]
    lab_keys  = [k for k, v in d.items() if _is_label_vector(v)]
    probs = {k: d[k] for k in prob_keys}
    labs  = {k: d[k] for k in lab_keys}

    def match_prob_label(Pcand, ycand):
        if Pcand is None:
            return None
        if ycand is not None and len(ycand) == Pcand.shape[0]:
            return Pcand, ycand
        for _, yv in labs.items():
            if len(yv) == Pcand.shape[0]:
                return Pcand, yv
        return None

    splits = {}
    m = match_prob_label(P_sel, y_sel)
    if m is not None:
        splits["sel"] = m
    m = match_prob_label(P_cal, y_cal)
    if m is not None:
        splits["cal"] = m
    m = match_prob_label(P_test, y_test)
    if m is not None:
        splits["test"] = m

    if "cal" not in splits:
        raise RuntimeError("Could not find calibration split in npz. Need probs+labels pair for cal.")
    if "test" not in splits:
        # fallback: split cal
        P, y = splits["cal"]
        rng = np.random.default_rng(fallback_split_seed)
        n = P.shape[0]
        idx = rng.permutation(n)
        n_cal = n // 2
        splits["cal"]  = (P[idx[:n_cal]], y[idx[:n_cal]])
        splits["test"] = (P[idx[n_cal:]], y[idx[n_cal:]])

    # normalize
    for k in list(splits.keys()):
        P, y = splits[k]
        P = _normalize_rows(P.astype(np.float64))
        y = y.astype(int)
        splits[k] = (P, y)

    return splits


# ---------------------------
# Conformal quantile
# ----------------------------
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = scores.size
    if n == 0:
        return 1.0
    q = math.ceil((n + 1) * (1 - alpha)) / n
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(scores, q, method="higher"))


# ---------------------------
# Vectorized APS / RAPS / SCC-RAPS
# ----------------------------
def _argsort_desc(P: np.ndarray) -> np.ndarray:
    return np.argsort(-P, axis=1)

def _inverse_perm(order: np.ndarray) -> np.ndarray:
    # order: (n,K)
    n, K = order.shape
    inv = np.empty_like(order)
    inv[np.arange(n)[:, None], order] = np.arange(K)[None, :]
    return inv

def scores_APS(P: np.ndarray, y: np.ndarray, randomized: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, K = P.shape
    order = _argsort_desc(P)
    P_sorted = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(P_sorted, axis=1)

    inv = _inverse_perm(order)
    r = inv[np.arange(n), y]  # 0-based rank index
    cum_y = cum[np.arange(n), r]
    if randomized:
        u = rng.uniform(0.0, 1.0, size=n)
        p_y = P_sorted[np.arange(n), r]
        cum_y = cum_y - u * p_y
    return cum_y

def scores_RAPS(P: np.ndarray, y: np.ndarray, lam: float, k_reg: int, randomized: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, K = P.shape
    order = _argsort_desc(P)
    P_sorted = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(P_sorted, axis=1)

    inv = _inverse_perm(order)
    r = inv[np.arange(n), y]  # 0-based
    L = r + 1                 # 1-based
    cum_y = cum[np.arange(n), r]
    if randomized:
        u = rng.uniform(0.0, 1.0, size=n)
        p_y = P_sorted[np.arange(n), r]
        cum_y = cum_y - u * p_y

    pen = lam * np.maximum(0, L - k_reg)
    return cum_y + pen

def scores_SCC_RAPS(
    P: np.ndarray,
    y: np.ndarray,
    class2cluster: np.ndarray,
    lam_c: np.ndarray,
    k_c: np.ndarray,
    randomized: bool,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, K = P.shape
    order = _argsort_desc(P)
    P_sorted = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(P_sorted, axis=1)

    inv = _inverse_perm(order)
    r = inv[np.arange(n), y]
    L = r + 1
    cum_y = cum[np.arange(n), r]
    if randomized:
        u = rng.uniform(0.0, 1.0, size=n)
        p_y = P_sorted[np.arange(n), r]
        cum_y = cum_y - u * p_y

    cl = class2cluster[y]
    lam = lam_c[cl]
    kk  = k_c[cl]
    pen = lam * np.maximum(0, L - kk)
    return cum_y + pen


def predsets_APS(P: np.ndarray, tau: float, randomized: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, K = P.shape
    order = _argsort_desc(P)
    P_sorted = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(P_sorted, axis=1)

    if randomized:
        u = rng.uniform(0.0, 1.0, size=n)[:, None]
        lhs = cum - u * P_sorted
    else:
        lhs = cum

    keep_sorted = lhs <= tau
    S = np.zeros((n, K), dtype=bool)
    S[np.arange(n)[:, None], order] = keep_sorted
    return S

def predsets_RAPS(P: np.ndarray, tau: float, lam: float, k_reg: int, randomized: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, K = P.shape
    order = _argsort_desc(P)
    P_sorted = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(P_sorted, axis=1)
    ranks = (np.arange(1, K + 1)[None, :]).astype(np.float64)
    pen = lam * np.maximum(0, ranks - k_reg)

    if randomized:
        u = rng.uniform(0.0, 1.0, size=n)[:, None]
        lhs = cum - u * P_sorted + pen
    else:
        lhs = cum + pen

    keep_sorted = lhs <= tau
    S = np.zeros((n, K), dtype=bool)
    S[np.arange(n)[:, None], order] = keep_sorted
    return S

def predsets_SCC_RAPS(
    P: np.ndarray,
    tau: float,
    class2cluster: np.ndarray,
    lam_c: np.ndarray,
    k_c: np.ndarray,
    randomized: bool,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, K = P.shape
    order = _argsort_desc(P)
    P_sorted = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(P_sorted, axis=1)
    ranks = (np.arange(1, K + 1)[None, :]).astype(np.float64)

    cl_sorted = class2cluster[order]          # (n,K)
    lam_sorted = lam_c[cl_sorted]             # (n,K)
    k_sorted   = k_c[cl_sorted]               # (n,K)
    pen = lam_sorted * np.maximum(0, ranks - k_sorted)

    if randomized:
        u = rng.uniform(0.0, 1.0, size=n)[:, None]
        lhs = cum - u * P_sorted + pen
    else:
        lhs = cum + pen

    keep_sorted = lhs <= tau
    S = np.zeros((n, K), dtype=bool)
    S[np.arange(n)[:, None], order] = keep_sorted
    return S


# ---------------------------
# Metrics
# ----------------------------
def evaluate_sets(S: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    n = S.shape[0]
    hit = S[np.arange(n), y].astype(float)
    sizes = S.sum(axis=1).astype(float)
    return {
        "coverage": float(hit.mean()),
        "avg_size": float(sizes.mean()),
        "p90_size": float(np.quantile(sizes, 0.90)),
        "p99_size": float(np.quantile(sizes, 0.99)),
    }


# ---------------------------
# Clustering: low-d class embedding from sel via random projection
# ----------------------------
def build_class2cluster_from_sel(
    P_sel: np.ndarray,
    y_sel: np.ndarray,
    K: int,
    C: int,
    emb_dim: int,
    seed: int,
) -> np.ndarray:
    """
    Build class embeddings:
      z_i = P_sel[i] @ R  where R is (K, emb_dim) random Rademacher / sqrt(emb_dim)
      emb[k] = mean_{i: y_sel[i]=k} z_i
    Then KMeans in R^{emb_dim} to cluster classes.
    """
    if C <= 1:
        return np.zeros(K, dtype=int)

    if KMeans is None:
        raise RuntimeError("scikit-learn not available. Install scikit-learn or set --clusters 1.")

    rng = np.random.default_rng(seed)
    # Rademacher projection
    R = rng.choice([-1.0, 1.0], size=(K, emb_dim)).astype(np.float32) / math.sqrt(emb_dim)

    Z = (P_sel.astype(np.float32) @ R)  # (n_sel, emb_dim)

    emb = np.zeros((K, emb_dim), dtype=np.float32)
    cnt = np.zeros(K, dtype=np.int64)
    for i in range(Z.shape[0]):
        k = int(y_sel[i])
        emb[k] += Z[i]
        cnt[k] += 1
    # normalize means; if unseen in sel -> small random / zeros
    for k in range(K):
        if cnt[k] > 0:
            emb[k] /= float(cnt[k])
        else:
            emb[k] = 0.0

    km = KMeans(n_clusters=C, random_state=seed, n_init="auto")
    c = km.fit_predict(emb)
    return c.astype(int)

def cluster_reliability_weights(P_sel: np.ndarray, y_sel: np.ndarray, class2cluster: np.ndarray) -> np.ndarray:
    """
    w_c = 1 - acc_c (harder cluster -> larger weight), normalized to mean 1.
    Uses sel split as "tune" proxy.
    """
    K = P_sel.shape[1]
    C = int(class2cluster.max()) + 1
    yhat = P_sel.argmax(axis=1)

    correct = np.zeros(C, dtype=np.float64)
    total   = np.zeros(C, dtype=np.float64)

    for i in range(P_sel.shape[0]):
        y = int(y_sel[i])
        c = int(class2cluster[y])
        total[c] += 1.0
        correct[c] += 1.0 if int(yhat[i]) == y else 0.0

    acc = correct / np.maximum(total, 1.0)
    w = 1.0 - acc
    w = np.maximum(w, 0.05)  # floor
    w = w / np.mean(w)
    return w


# ---------------------------
# Main
# ----------------------------
@dataclass
class Row:
    method: str
    coverage: float
    avg_size: float
    p90_size: float
    p99_size: float

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=0.1)

    # randomized APS/RAPS
    ap.add_argument("--randomized", action="store_true")

    # RAPS params
    ap.add_argument("--lambda_raps", type=float, default=0.01)
    ap.add_argument("--k_reg", type=int, default=5)

    # SCC-RAPS params
    ap.add_argument("--clusters", type=int, default=10)
    ap.add_argument("--emb_dim", type=int, default=256, help="dim for random-projection class embedding")
    ap.add_argument("--lambda0", type=float, default=0.01)
    ap.add_argument("--shrinkage", action="store_true", help="lambda_c = lambda0 * w_c(reliability)")
    ap.add_argument("--k_same", action="store_true", help="use same k_reg for all clusters (recommended)")

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="", help="optional json output path")

    args = ap.parse_args()

    splits = load_npz_splits(args.npz, K=args.K, fallback_split_seed=args.seed)
    P_cal, y_cal = splits["cal"]
    P_test, y_test = splits["test"]
    if "sel" in splits:
        P_sel, y_sel = splits["sel"]
    else:
        # not ideal; but keep runnable
        P_sel, y_sel = P_cal, y_cal

    K = args.K
    print(f"[file] {args.npz}")
    print(f"[splits] sel={P_sel.shape} cal={P_cal.shape} test={P_test.shape}  K={K}")
    print(f"[alpha] {args.alpha}  [randomized] {args.randomized}")

    # --- clusters ---
    class2cluster = build_class2cluster_from_sel(
        P_sel=P_sel, y_sel=y_sel, K=K,
        C=args.clusters, emb_dim=args.emb_dim, seed=args.seed
    )
    C = int(class2cluster.max()) + 1
    print(f"[cluster] C={C}  (requested {args.clusters})")

    if args.k_same:
        k_c = np.full(C, args.k_reg, dtype=int)
    else:
        k_c = np.full(C, args.k_reg, dtype=int)

    if args.shrinkage and C > 1:
        w = cluster_reliability_weights(P_sel, y_sel, class2cluster)
        lam_c = args.lambda0 * w
        print(f"[scc] shrinkage ON: lambda0={args.lambda0}  w(min/mean/max)=({w.min():.3f},{w.mean():.3f},{w.max():.3f})")
    else:
        lam_c = np.full(C, args.lambda0, dtype=float)

    # --- calibrate taus on cal ---
    E_aps  = scores_APS(P_cal, y_cal, randomized=args.randomized, seed=args.seed + 10)
    tau_aps = conformal_quantile(E_aps, args.alpha)

    E_raps = scores_RAPS(P_cal, y_cal, lam=args.lambda_raps, k_reg=args.k_reg,
                         randomized=args.randomized, seed=args.seed + 20)
    tau_raps = conformal_quantile(E_raps, args.alpha)

    E_scc  = scores_SCC_RAPS(P_cal, y_cal, class2cluster, lam_c, k_c,
                             randomized=args.randomized, seed=args.seed + 30)
    tau_scc = conformal_quantile(E_scc, args.alpha)

    print(f"[tau] APS={tau_aps:.6f} | RAPS={tau_raps:.6f} | SCC-RAPS={tau_scc:.6f}")

    # --- build sets on test ---
    S_aps  = predsets_APS(P_test, tau_aps, randomized=args.randomized, seed=args.seed + 101)
    S_raps = predsets_RAPS(P_test, tau_raps, lam=args.lambda_raps, k_reg=args.k_reg,
                           randomized=args.randomized, seed=args.seed + 102)
    S_scc  = predsets_SCC_RAPS(P_test, tau_scc, class2cluster, lam_c, k_c,
                               randomized=args.randomized, seed=args.seed + 103)

    r_aps  = evaluate_sets(S_aps,  y_test)
    r_raps = evaluate_sets(S_raps, y_test)
    r_scc  = evaluate_sets(S_scc,  y_test)

    rows = [
        Row("APS", **r_aps),
        Row("RAPS", **r_raps),
        Row("SCC-RAPS", **r_scc),
    ]

    print("")
    print(f"{'method':8s} | {'cov':>7s} | {'size':>8s} | {'p90':>8s} | {'p99':>8s}")
    print("-" * 50)
    for rr in rows:
        print(f"{rr.method:8s} | {rr.coverage:7.4f} | {rr.avg_size:8.3f} | {rr.p90_size:8.1f} | {rr.p99_size:8.1f}")

    if args.out:
        out = {
            "npz": args.npz,
            "K": K,
            "alpha": args.alpha,
            "seed": args.seed,
            "randomized": args.randomized,
            "clusters": int(C),
            "emb_dim": int(args.emb_dim),
            "tau": {"APS": tau_aps, "RAPS": tau_raps, "SCC_RAPS": tau_scc},
            "RAPS_params": {"lambda": float(args.lambda_raps), "k_reg": int(args.k_reg)},
            "SCC_params": {
                "lambda0": float(args.lambda0),
                "lambda_c": lam_c.tolist(),
                "k_c": k_c.tolist(),
            },
            "test": {"APS": r_aps, "RAPS": r_raps, "SCC_RAPS": r_scc},
            "class2cluster": class2cluster.tolist(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
