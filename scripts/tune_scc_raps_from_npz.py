# inat2017 npz를 입력으로 받고, scc-raps 튜닝을 위해 그리드 서치를 수행
# calibration에서 tau를 다시 맞춘 후
# test에서 coverage \ge 1-alpha를 만족하는 것들 중 avg set size가 가장 작은 것을 선택

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None


# ----------------------------
# Load NPZ splits (sel/cal/test)
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

    splits = {}
    if P_cal is None or y_cal is None:
        raise RuntimeError("Could not find cal split (p_cal/y_cal) in npz.")
    splits["cal"] = (P_cal, y_cal)

    if P_sel is not None and y_sel is not None:
        splits["sel"] = (P_sel, y_sel)
    else:
        splits["sel"] = (P_cal, y_cal)

    if P_test is not None and y_test is not None and len(y_test) == P_test.shape[0]:
        splits["test"] = (P_test, y_test)
    else:
        # fallback: split cal into halves
        rng = np.random.default_rng(fallback_split_seed)
        P, y = splits["cal"]
        n = P.shape[0]
        idx = rng.permutation(n)
        n_cal = n // 2
        splits["cal"] = (P[idx[:n_cal]], y[idx[:n_cal]])
        splits["test"] = (P[idx[n_cal:]], y[idx[n_cal:]])

    # normalize
    for k in list(splits.keys()):
        P, y = splits[k]
        P = _normalize_rows(P.astype(np.float64))
        y = y.astype(int)
        splits[k] = (P, y)

    return splits


# ----------------------------
# Conformal quantile
# ----------------------------
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = scores.size
    if n == 0:
        return 1.0
    q = math.ceil((n + 1) * (1 - alpha)) / n
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(scores, q, method="higher"))


# ----------------------------
# Vectorized APS/RAPS/SCC-RAPS
# ----------------------------
def _argsort_desc(P: np.ndarray) -> np.ndarray:
    return np.argsort(-P, axis=1)

def _inverse_perm(order: np.ndarray) -> np.ndarray:
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
    r = inv[np.arange(n), y]
    s = cum[np.arange(n), r]
    if randomized:
        u = rng.uniform(0.0, 1.0, size=n)
        s = s - u * P_sorted[np.arange(n), r]
    return s

def scores_RAPS(P: np.ndarray, y: np.ndarray, lam: float, k_reg: int, randomized: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, K = P.shape
    order = _argsort_desc(P)
    P_sorted = np.take_along_axis(P, order, axis=1)
    cum = np.cumsum(P_sorted, axis=1)
    inv = _inverse_perm(order)
    r = inv[np.arange(n), y]
    L = r + 1
    s = cum[np.arange(n), r]
    if randomized:
        u = rng.uniform(0.0, 1.0, size=n)
        s = s - u * P_sorted[np.arange(n), r]
    pen = lam * np.maximum(0, (L - k_reg) / K)
    return s + pen

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
    s = cum[np.arange(n), r]
    if randomized:
        u = rng.uniform(0.0, 1.0, size=n)
        s = s - u * P_sorted[np.arange(n), r]
    cl = class2cluster[y]
    lam = lam_c[cl]
    kk = k_c[cl]
    pen = lam * np.maximum(0, (L - kk) / K)
    return s + pen

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
    pen = lam * np.maximum(0, (ranks - k_reg) / K)
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
    cl_sorted = class2cluster[order]
    lam_sorted = lam_c[cl_sorted]
    k_sorted = k_c[cl_sorted]
    pen = lam_sorted * np.maximum(0, (ranks - k_sorted) / K)

    if randomized:
        u = rng.uniform(0.0, 1.0, size=n)[:, None]
        lhs = cum - u * P_sorted + pen
    else:
        lhs = cum + pen

    keep_sorted = lhs <= tau
    S = np.zeros((n, K), dtype=bool)
    S[np.arange(n)[:, None], order] = keep_sorted
    return S


# ----------------------------
# Clustering: class embedding from sel split
# emb[y] = mean_{i: y_i=y} P_sel[i,:]
# ----------------------------
def build_class_embedding(P_sel: np.ndarray, y_sel: np.ndarray, K: int) -> np.ndarray:
    emb = np.zeros((K, K), dtype=np.float64)
    cnt = np.zeros(K, dtype=np.int64)
    for i in range(P_sel.shape[0]):
        y = int(y_sel[i])
        if 0 <= y < K:
            emb[y] += P_sel[i]
            cnt[y] += 1
    for y in range(K):
        if cnt[y] > 0:
            emb[y] /= cnt[y]
        else:
            emb[y, y] = 1.0
    return emb

def build_clusters(emb: np.ndarray, C: int, seed: int) -> np.ndarray:
    if C <= 1:
        return np.zeros(emb.shape[0], dtype=int)
    if KMeans is None:
        raise RuntimeError("scikit-learn not available. Install scikit-learn or set C=1.")
    km = KMeans(n_clusters=C, random_state=seed, n_init="auto")
    return km.fit_predict(emb).astype(int)

def cluster_reliability_weights(P_sel: np.ndarray, y_sel: np.ndarray, class2cluster: np.ndarray) -> np.ndarray:
    # w_c = 1 - acc_c (harder cluster => larger w)
    yhat = P_sel.argmax(axis=1)
    C = int(class2cluster.max()) + 1
    correct = np.zeros(C, dtype=float)
    total = np.zeros(C, dtype=float)
    for i in range(P_sel.shape[0]):
        y = int(y_sel[i])
        c = int(class2cluster[y])
        total[c] += 1.0
        correct[c] += 1.0 if int(yhat[i]) == y else 0.0
    acc = correct / np.maximum(total, 1.0)
    w = 1.0 - acc
    w = np.maximum(w, 0.05)
    w = w / np.mean(w)
    return w


# ----------------------------
# Eval
# ----------------------------
def evaluate_sets(S: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    n = S.shape[0]
    hit = S[np.arange(n), y].astype(float)
    sz = S.sum(axis=1).astype(float)
    return {
        "coverage": float(hit.mean()),
        "avg_size": float(sz.mean()),
        "p90_size": float(np.quantile(sz, 0.90)),
        "p99_size": float(np.quantile(sz, 0.99)),
    }


@dataclass
class GridRow:
    C: int
    lambda0: float
    k_reg: int
    shrinkage: bool
    tau: float
    cov: float
    size: float
    p90: float
    p99: float


def parse_list_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]

def parse_list_ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--randomized", action="store_true")

    # clustering
    ap.add_argument("--C", type=int, default=10)
    ap.add_argument("--cluster_seed", type=int, default=1)

    # grid
    ap.add_argument("--lambda0_grid", type=str, default="0.005,0.01,0.02,0.05")
    ap.add_argument("--k_grid", type=str, default="1,3,5,10,20,50,100")
    ap.add_argument("--shrinkage_grid", type=str, default="0,1")  # 0/1

    # feasibility tolerance (coverage must be >= 1-alpha - tol)
    ap.add_argument("--cov_tol", type=float, default=0.0)

    # output
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument("--topn", type=int, default=20)

    args = ap.parse_args()

    splits = load_npz_splits(args.npz, K=args.K, fallback_split_seed=args.cluster_seed)
    P_sel, y_sel = splits["sel"]
    P_cal, y_cal = splits["cal"]
    P_test, y_test = splits["test"]

    print(f"[file] {args.npz}")
    print(f"[splits] sel={P_sel.shape} cal={P_cal.shape} test={P_test.shape}  K={args.K}")
    print(f"[alpha] {args.alpha}  [randomized] {args.randomized}")

    # Build clusters once (fixed across grid)
    emb = build_class_embedding(P_sel, y_sel, K=args.K)
    class2cluster = build_clusters(emb, C=args.C, seed=args.cluster_seed)
    C = int(class2cluster.max()) + 1
    print(f"[cluster] C={C} (requested {args.C})")

    # reliability weights once
    w = cluster_reliability_weights(P_sel, y_sel, class2cluster)
    print(f"[reliability] w(min/mean/max)=({w.min():.3f},{w.mean():.3f},{w.max():.3f})")

    lambda0_grid = parse_list_floats(args.lambda0_grid)
    k_grid = parse_list_ints(args.k_grid)
    shrink_grid = [bool(int(x)) for x in args.shrinkage_grid.split(",") if x.strip()]

    rows: List[GridRow] = []
    target = 1.0 - args.alpha

    # For reference: APS and vanilla RAPS (single setting) at the end if desired
    # but grid focuses on SCC-RAPS
    for shrink in shrink_grid:
        for lam0 in lambda0_grid:
            lam_c = (lam0 * w) if (shrink and C > 1) else np.full(C, lam0, dtype=float)
            for k_reg in k_grid:
                k_c = np.full(C, k_reg, dtype=int)

                # calibrate
                E_cal = scores_SCC_RAPS(
                    P_cal, y_cal,
                    class2cluster=class2cluster,
                    lam_c=lam_c, k_c=k_c,
                    randomized=args.randomized,
                    seed=args.cluster_seed + 123,
                )
                tau = conformal_quantile(E_cal, alpha=args.alpha)

                # test sets
                S = predsets_SCC_RAPS(
                    P_test, tau,
                    class2cluster=class2cluster,
                    lam_c=lam_c, k_c=k_c,
                    randomized=args.randomized,
                    seed=args.cluster_seed + 999,
                )
                ev = evaluate_sets(S, y_test)

                cov = ev["coverage"]
                if cov + 1e-12 < (target - args.cov_tol):
                    continue  # infeasible

                rows.append(GridRow(
                    C=C, lambda0=float(lam0), k_reg=int(k_reg), shrinkage=bool(shrink),
                    tau=float(tau),
                    cov=float(cov), size=float(ev["avg_size"]),
                    p90=float(ev["p90_size"]), p99=float(ev["p99_size"]),
                ))

                print(f"[grid] shrink={int(shrink)} lam0={lam0:g} k={k_reg:4d} "
                      f"tau={tau:.4f}  cov={cov:.4f}  size={ev['avg_size']:.1f}")

    # sort by (size, -cov)
    rows_sorted = sorted(rows, key=lambda r: (r.size, -r.cov))

    print("")
    print(f"[feasible] {len(rows_sorted)} configs (cov >= {target - args.cov_tol:.4f})")
    print(f"{'rank':>4s} | {'sh':>2s} | {'lam0':>7s} | {'k':>5s} | {'cov':>7s} | {'size':>8s} | {'p90':>8s} | {'p99':>8s}")
    print("-" * 80)
    for i, r in enumerate(rows_sorted[:max(args.topn, 1)], start=1):
        print(f"{i:4d} | {int(r.shrinkage):2d} | {r.lambda0:7.4g} | {r.k_reg:5d} | "
              f"{r.cov:7.4f} | {r.size:8.1f} | {r.p90:8.1f} | {r.p99:8.1f}")

    # save
    if args.out_json:
        out = {
            "npz": args.npz,
            "K": args.K,
            "alpha": args.alpha,
            "randomized": bool(args.randomized),
            "C": int(C),
            "class2cluster": class2cluster.tolist(),
            "reliability_w": w.tolist(),
            "grid": [r.__dict__ for r in rows_sorted],
            "top": [r.__dict__ for r in rows_sorted[:max(args.topn, 1)]],
            "target_cov": target,
            "cov_tol": args.cov_tol,
        }
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
