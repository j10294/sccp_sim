import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

#----------------------------
# CCCP Utilities #
#----------------------------
import numpy as np
from typing import Optional, Tuple

def min_count_for_null(alpha: float) -> int:
    """
    CCCP-style null-cluster criterion:
      if n_y < (1 / min(alpha, 0.1)) - 1  => null cluster
    """
    thr = (1.0 / min(alpha, 0.1)) - 1.0
    # "보다 작으면" 이므로 정수화는 floor 성격이 자연스러움
    return int(np.floor(thr))

def weighted_kmeans_simple(
    X: np.ndarray,
    n_clusters: int,
    weights: np.ndarray,
    seed: int = 1,
    n_iter: int = 50,
) -> np.ndarray:
    """
    Minimal weighted k-means:
      - assignment: Euclidean distance
      - update: weighted mean with weights
    """
    rng = np.random.default_rng(seed)
    N, d = X.shape
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0.0, None)

    # init centers: sample points proportional to weights (fallback to uniform)
    if w.sum() > 0:
        p = w / w.sum()
        init_idx = rng.choice(N, size=n_clusters, replace=False, p=p)
    else:
        init_idx = rng.choice(N, size=n_clusters, replace=False)

    centers = X[init_idx].copy()
    labels = np.zeros(N, dtype=int)

    for _ in range(n_iter):
        # assign
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels

        # update (weighted mean)
        for c in range(n_clusters):
            m = (labels == c)
            if not m.any():
                # re-init empty cluster
                if w.sum() > 0:
                    idx = rng.choice(N, p=w / w.sum())
                else:
                    idx = rng.integers(0, N)
                centers[c] = X[idx]
                continue

            wm = w[m]
            if wm.sum() <= 0:
                centers[c] = X[m].mean(axis=0)
            else:
                centers[c] = (X[m] * wm[:, None]).sum(axis=0) / wm.sum()

    return labels

def class_quantile_embedding_with_null(
    scores_sel: np.ndarray,
    y_sel: np.ndarray,
    K: int,
    q_grid: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      emb: (K, d) quantile embedding (global quantiles used as filler for non-null only if needed)
      n_y: (K,) counts in sel
      is_null: (K,) boolean indicating null-cluster classes
    """
    q_grid = np.asarray(q_grid, dtype=float)
    d = len(q_grid)

    y_sel = np.asarray(y_sel, dtype=int)
    scores_sel = np.asarray(scores_sel, dtype=float)

    n_y = np.bincount(y_sel, minlength=K)
    n_min = min_count_for_null(alpha)
    is_null = n_y < n_min  # strict "<" as you specified

    # global quantiles (for numerical safety / fallback)
    global_q = np.quantile(scores_sel, q_grid, method="higher")

    emb = np.zeros((K, d), dtype=float)
    for k in range(K):
        mk = (y_sel == k)
        if mk.any() and (not is_null[k]):
            emb[k] = np.quantile(scores_sel[mk], q_grid, method="higher")
        else:
            # null classes: embedding value won't be used in clustering
            # but fill with global_q to avoid NaNs
            emb[k] = global_q

    # numerical safety
    bad = ~np.isfinite(emb)
    if bad.any():
        emb[bad] = np.take(global_q, np.where(bad)[1])

    return emb, n_y.astype(int), is_null

def cluster_classes_cccp_style(
    scores_sel: np.ndarray,
    y_sel: np.ndarray,
    K: int,
    q_grid: np.ndarray,
    alpha: float,
    n_clusters: int,
    seed: int,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    CCCP-style clustering:
      1) null cluster assignment by n_y < (1/min(alpha,0.1))-1
      2) run weighted k-means on remaining classes with weight sqrt(n_y)
      3) return class2cluster where null classes map to an extra cluster index (n_clusters)
    
    Returns:
      class2cluster: (K,) in {0,...,n_clusters}  (last id = null cluster)
      C_eff: n_clusters + 1
      is_null: (K,)
    """
    emb, n_y, is_null = class_quantile_embedding_with_null(
        scores_sel=scores_sel,
        y_sel=y_sel,
        K=K,
        q_grid=q_grid,
        alpha=alpha,
    )

    idx_nonnull = np.where(~is_null)[0]
    idx_null = np.where(is_null)[0]

    # If all are null (extreme case), everything goes to null cluster
    if len(idx_nonnull) == 0:
        class2cluster = np.full(K, n_clusters, dtype=int)
        return class2cluster, n_clusters + 1, is_null

    X = emb[idx_nonnull]
    w = np.sqrt(np.maximum(n_y[idx_nonnull], 0)).astype(float)

    # If fewer non-null classes than clusters, reduce clusters
    Kc = min(n_clusters, len(idx_nonnull))
    labels_nonnull = weighted_kmeans_simple(X, n_clusters=Kc, weights=w, seed=seed)

    # Map back to full K, and reserve last cluster id = null cluster
    class2cluster = np.full(K, n_clusters, dtype=int)  # default null cluster id
    class2cluster[idx_nonnull] = labels_nonnull

    # Effective cluster count:
    # - non-null clusters: 0..Kc-1
    # - (optional) unused cluster ids if Kc<n_clusters are fine but we will compress to Kc
    # - null cluster id: Kc
    # To keep things clean, compress cluster ids to 0..Kc-1 then null=Kc
    # (This avoids "max cluster id" mismatches later.)
    # relabel non-null cluster ids to 0..Kc-1 already; set null to Kc
    class2cluster[idx_null] = Kc
    C_eff = Kc + 1

    return class2cluster, C_eff, is_null

# ----------------------------
# Utilities: load & auto-detect arrays
# ----------------------------
def _is_prob_matrix(a: np.ndarray, K: int) -> bool:
    return isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == K and np.isfinite(a).all()

def _is_label_vector(a: np.ndarray) -> bool:
    return isinstance(a, np.ndarray) and a.ndim == 1 and np.issubdtype(a.dtype, np.integer)

def _normalize_rows(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # Ensure row sums are 1 (robust to minor drift)
    s = P.sum(axis=1, keepdims=True)
    s = np.where(s <= eps, 1.0, s)
    Pn = P / s
    Pn = np.clip(Pn, eps, 1.0)
    Pn = Pn / Pn.sum(axis=1, keepdims=True)
    return Pn

def _find_by_name(d: Dict[str, np.ndarray], include: Tuple[str, ...], K: int) -> Optional[np.ndarray]:
    # find prob matrices by substring in key name
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

def load_npz_splits(path: str, K: int = 100, fallback_split_seed: int = 1) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Returns dict with keys among {'sel','cal','test'} mapping to (P, y).
    Auto-detect common key patterns; fallback if test missing.
    """
    raw = np.load(path, allow_pickle=True)
    d = {k: raw[k] for k in raw.files}
        # ---- Common patterns (SAFE: no `or` on numpy arrays) ----
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

    # If not found by name, try heuristic pairing:
    # Collect all prob matrices and label vectors; match by length.
    prob_keys = [k for k, v in d.items() if _is_prob_matrix(v, K)]
    lab_keys = [k for k, v in d.items() if _is_label_vector(v)]
    probs = {k: d[k] for k in prob_keys}
    labs = {k: d[k] for k in lab_keys}

    def match_prob_label(Pcand: Optional[np.ndarray], ycand: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if Pcand is None:
            return None
        if ycand is not None and len(ycand) == Pcand.shape[0]:
            return Pcand, ycand
        # try find any label with same length
        for _, yv in labs.items():
            if len(yv) == Pcand.shape[0]:
                return Pcand, yv
        return None

    splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    m = match_prob_label(P_sel, y_sel)
    if m is not None:
        splits["sel"] = m
    m = match_prob_label(P_cal, y_cal)
    if m is not None:
        splits["cal"] = m
    m = match_prob_label(P_test, y_test)
    if m is not None:
        splits["test"] = m

    # If still missing, attempt to assign largest as train (ignore), next as cal/test:
    if "cal" not in splits:
        # choose a prob matrix that has a matching label and is not the same object as sel/test
        candidates = []
        for pk, Pv in probs.items():
            my = None
            for yk, yv in labs.items():
                if len(yv) == Pv.shape[0]:
                    my = yv
                    break
            if my is not None:
                candidates.append((pk, Pv, my))
        # sort by n desc
        candidates.sort(key=lambda t: t[1].shape[0], reverse=True)
        # heuristic: if we have 3 sets, biggest=train, then cal, then test
        if len(candidates) >= 2:
            # take second as cal
            splits["cal"] = (candidates[1][1], candidates[1][2])
        elif len(candidates) == 1:
            splits["cal"] = (candidates[0][1], candidates[0][2])

    if "test" not in splits:
        # fallback: split cal into cal/test halves
        if "cal" not in splits:
            raise RuntimeError("Could not find calibration split in npz. Need at least one probs+labels pair.")
        P, y = splits["cal"]
        rng = np.random.default_rng(fallback_split_seed)
        n = P.shape[0]
        idx = rng.permutation(n)
        n_cal = n // 2
        cal_idx = idx[:n_cal]
        test_idx = idx[n_cal:]
        splits["cal"] = (P[cal_idx], y[cal_idx])
        splits["test"] = (P[test_idx], y[test_idx])

    # Normalize probabilities
    for k in list(splits.keys()):
        P, y = splits[k]
        P = _normalize_rows(P.astype(np.float64))
        y = y.astype(int)
        splits[k] = (P, y)

    return splits

# ----------------------------
# Utilities : tail/head index metrics
# ----------------------------
import numpy as np

def split_tail_head(K: int, tail_set: np.ndarray):
    tail_set = np.asarray(tail_set, dtype=int)
    tail_mask = np.zeros(K, dtype=bool)
    tail_mask[tail_set] = True
    head_mask = ~tail_mask
    tail_idx = np.where(tail_mask)[0]
    head_idx = np.where(head_mask)[0]
    return tail_idx, head_idx

def summarize(arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    return {
        "avg": float(np.nanmean(arr)),
        "worst": float(np.nanmin(arr)),
        "std": float(np.nanstd(arr)),
    }

def report_tail_head(method_name: str, cls_cov: np.ndarray, tail_idx: np.ndarray, head_idx: np.ndarray):
    tail_sum = summarize(cls_cov[tail_idx]) if len(tail_idx) else None
    head_sum = summarize(cls_cov[head_idx]) if len(head_idx) else None

    print(f"\n[{method_name}] classwise coverage split")
    if tail_sum is not None:
        print(f"  tail (m={len(tail_idx)}): avg={tail_sum['avg']:.4f} worst={tail_sum['worst']:.4f} std={tail_sum['std']:.4f}")
    if head_sum is not None:
        print(f"  head (m={len(head_idx)}): avg={head_sum['avg']:.4f} worst={head_sum['worst']:.4f} std={head_sum['std']:.4f}")

#----------------------------
# top K 출력
#----------------------------

def true_label_rank(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    rank_true[i] = 1 means true label is top-1 for sample i.
    """
    P = np.asarray(P)
    y = np.asarray(y).astype(int)
    order = np.argsort(-P, axis=1)  # descending
    pos = (order == y[:, None]).argmax(axis=1)
    return pos + 1

def summarize_topM(P: np.ndarray, y: np.ndarray, Ms=(1, 5, 10, 20, 50, 100, 200, 500, 1000)) -> dict:
    r = true_label_rank(P, y)
    out = {}
    out["n"] = int(len(r))
    out["rank_mean"] = float(np.mean(r))
    out["rank_quantiles"] = {
        "q50": float(np.quantile(r, 0.50)),
        "q90": float(np.quantile(r, 0.90)),
        "q95": float(np.quantile(r, 0.95)),
        "q99": float(np.quantile(r, 0.99)),
    }
    out["topM_acc"] = {int(M): float(np.mean(r <= M)) for M in Ms}
    return out

def _parse_int_list(s: str) -> list:
    # e.g. "1,5,10,50,100"
    xs = []
    for t in s.split(","):
        t = t.strip()
        if t:
            xs.append(int(t))
    xs = sorted(set(xs))
    return xs

# ----------------------------
# Conformal methods
# ----------------------------
def score_s1(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Nonconformity score: s = 1 - p_true (smaller is better)
    return 1.0 - P[np.arange(P.shape[0]), y]

def quantile_upper(scores: np.ndarray, alpha: float) -> float:
    # For split conformal classification with score s, threshold is (1-alpha)-quantile.
    # Use conservative quantile: ceil((n+1)*(1-alpha))/n style.
    n = len(scores)
    if n == 0:
        return 1.0
    k = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(np.sort(scores)[k])

def predset_from_threshold(P: np.ndarray, t: float) -> np.ndarray:
    # include label j if 1 - P_ij <= t  <=>  P_ij >= 1 - t
    return P >= (1.0 - t)

def eval_sets(
    S: np.ndarray,
    y: np.ndarray,
    K: int,
    class2cluster: Optional[np.ndarray] = None,
    tail_set: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Returns overall, classwise, tail/head (optional), and (optional) clusterwise metrics.
    """
    n, K2 = S.shape
    assert K2 == K

    y = np.asarray(y, dtype=int)

    hit = S[np.arange(n), y].astype(float)
    size_i = S.sum(axis=1).astype(float)

    out: Dict[str, object] = {}
    out["coverage"] = float(np.mean(hit))
    out["avg_size"] = float(np.mean(size_i))

    # ---- tail/head (samplewise, by true label) ----
    if tail_set is not None:
        tail_set = np.asarray(tail_set, dtype=int)
        is_tail = np.isin(y, tail_set)
        is_head = ~is_tail

        out["n_tail"] = int(is_tail.sum())
        out["n_head"] = int(is_head.sum())

        out["cov_tail"] = float(np.mean(hit[is_tail])) if is_tail.any() else float("nan")
        out["cov_head"] = float(np.mean(hit[is_head])) if is_head.any() else float("nan")
        out["size_tail"] = float(np.mean(size_i[is_tail])) if is_tail.any() else float("nan")
        out["size_head"] = float(np.mean(size_i[is_head])) if is_head.any() else float("nan")

    # ---- classwise ----
    cov_k = np.full(K, np.nan, dtype=float)
    size_k = np.full(K, np.nan, dtype=float)
    n_k = np.zeros(K, dtype=int)

    for k in range(K):
        m = (y == k)
        n_k[k] = int(m.sum())
        if n_k[k] > 0:
            cov_k[k] = float(np.mean(hit[m]))
            size_k[k] = float(np.mean(size_i[m]))

    out["n_class"] = n_k.tolist()
    out["cov_class"] = cov_k.tolist()
    out["size_class"] = size_k.tolist()
    out["worst_class_cov"] = float(np.nanmin(cov_k))
    out["std_class_cov"] = float(np.nanstd(cov_k))
    out["avg_class_cov"] = float(np.nanmean(cov_k))

    # ---- clusterwise (if mapping provided) ----
    if class2cluster is not None:
        class2cluster = np.asarray(class2cluster, dtype=int)
        C = int(class2cluster.max()) + 1

        cov_c = np.full(C, np.nan, dtype=float)
        size_c = np.full(C, np.nan, dtype=float)
        n_c = np.zeros(C, dtype=int)

        y_cluster = class2cluster[y]

        for c in range(C):
            m = (y_cluster == c)
            n_c[c] = int(m.sum())
            if n_c[c] > 0:
                cov_c[c] = float(np.mean(hit[m]))
                size_c[c] = float(np.mean(size_i[m]))

        out["n_cluster"] = n_c.tolist()
        out["cov_cluster"] = cov_c.tolist()
        out["size_cluster"] = size_c.tolist()
        out["worst_cluster_cov"] = float(np.nanmin(cov_c))
        out["std_cluster_cov"] = float(np.nanstd(cov_c))
        out["avg_cluster_cov"] = float(np.nanmean(cov_c))

    return out

# ----------------------------
# SCCP: clustering classes + shrinkage of class-quantiles
# ----------------------------

import numpy as np

def class_quantile_embedding(
    scores_sel: np.ndarray,   # shape (n_sel,)
    y_sel: np.ndarray,        # shape (n_sel,)
    K: int,
    q_grid: np.ndarray,       # e.g., [0.5,0.6,0.7,0.8,0.9,1-alpha]
    fallback: str = "global", # or "nan_to_global"
) -> np.ndarray:
    """
    For each class k, embed it by quantiles of its score distribution on selection set:
        emb_k[j] = Quantile_{q_grid[j]}( scores_sel | y_sel=k ).
    If class has no selection samples, fallback to global quantiles.
    """
    q_grid = np.asarray(q_grid, dtype=np.float64)
    d = len(q_grid)

    emb = np.zeros((K, d), dtype=np.float64)

    # global fallback quantiles
    global_q = np.quantile(scores_sel, q_grid, method="higher")

    for k in range(K):
        m = (y_sel == k)
        if m.any():
            emb[k] = np.quantile(scores_sel[m], q_grid, method="higher")
        else:
            emb[k] = global_q

    # replace non-finite just in case
    bad = ~np.isfinite(emb)
    if bad.any():
        emb[bad] = np.take(global_q, np.where(bad)[1])

    return emb

def kmeans_simple(X: np.ndarray, n_clusters: int, seed: int = 1, n_iter: int = 50) -> np.ndarray:
    # Minimal k-means (no sklearn dependency)
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    # init centers by random points
    centers = X[rng.choice(N, size=n_clusters, replace=False)].copy()
    labels = np.zeros(N, dtype=int)
    for _ in range(n_iter):
        # assign
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # update
        for c in range(n_clusters):
            m = labels == c
            if m.any():
                centers[c] = X[m].mean(axis=0)
            else:
                centers[c] = X[rng.integers(0, N)]
    return labels

from typing import Tuple, Optional, List
import numpy as np

# -------------------------
# SCCP thresholds (global + cluster shrinkage)
# -------------------------
def sccp_thresholds(
    P_sel: np.ndarray,
    y_sel: np.ndarray,
    P_cal: np.ndarray,
    y_cal: np.ndarray,
    alpha: float,
    n_clusters: int = 10,
    shrink_tau: float = 50.0,
    seed: int = 1,
    embed_mode: str = "score_quantile",
    q_grid: Optional[List[float]] = None,
    cluster_mode : str = 'plain'
      # 'plain' or 'cccp_null'
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      - t_class: per-class thresholds (CCCP-like; reporting)
      - t_cluster: per-cluster thresholds
      - global_t: global threshold
      - t_mix_cluster: per-cluster mixed thresholds (cluster+global shrinkage)
      - t_sccp: per-class SCCP thresholds (inherit cluster mixed threshold)
      - c_labels: class->cluster mapping
    """
    K = P_cal.shape[1]

    scores_sel = score_s1(P_sel, y_sel)
    scores_cal = score_s1(P_cal, y_cal)
    global_t = quantile_upper(scores_cal, alpha)

    # --- embedding for clustering ---
    if embed_mode == "prob_mean":
        emb = np.zeros((K, K), dtype=np.float64)
        for k in range(K):
            m = (y_sel == k)
            emb[k] = P_sel[m].mean(axis=0) if m.any() else 0.0

    elif embed_mode == "score_quantile":
        if q_grid is None:
            q_grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0 - alpha]
        q_grid_arr = np.array(q_grid, dtype=np.float64)

        # quantile embedding (null 없이도 안정적으로 채우기 위해 global로 fallback)
        emb = class_quantile_embedding(
            scores_sel=scores_sel, y_sel=y_sel, K=K, q_grid=q_grid_arr, fallback="global"
        )
    else:
        raise ValueError(f"Unknown embed_mode={embed_mode}")

    # --- cluster classes: CCCP-style(null) OR plain k-means ---
    if cluster_mode == "cccp_null":
        if embed_mode != "score_quantile":
            raise ValueError("cluster_mode='cccp_null' requires embed_mode='score_quantile'")

        c_labels, C_eff, is_null = cluster_classes_cccp_style(
            scores_sel=scores_sel,
            y_sel=y_sel,
            K=K,
            q_grid=q_grid_arr,
            alpha=alpha,
            n_clusters=n_clusters,
            seed=seed,
        )

    elif cluster_mode == "plain" :
        c_labels = kmeans_simple(emb, n_clusters=n_clusters, seed=seed)
        C = int(n_clusters)
    else:
        raise ValueError(f"Unknown cluster_mode={cluster_mode}")


    # class thresholds (optional CCCP-like)
    t_class = np.full(K, np.nan, dtype=np.float64)
    for k in range(K):
        m = (y_cal == k)
        if m.any():
            t_class[k] = quantile_upper(scores_cal[m], alpha)

    
    # ------------------------------------------------------------
    # cluster thresholds (use effective cluster count!)
    # ------------------------------------------------------------
    C = int(C_eff) if ("C_eff" in locals() and C_eff is not None) else int(n_clusters)

    t_cluster = np.full(C, np.nan, dtype=np.float64)
    n_cluster = np.zeros(C, dtype=np.int64)

    for c in range(C):
        cls_in_c = np.where(c_labels == c)[0]
        m = np.isin(y_cal, cls_in_c)
        n_cluster[c] = int(m.sum())
        if m.any():
            t_cluster[c] = quantile_upper(scores_cal[m], alpha)
        else:
            t_cluster[c] = global_t

    # ------------------------------------------------------------
    # global + cluster shrinkage (per cluster)
    # ------------------------------------------------------------
    t_mix_cluster = np.zeros(C, dtype=np.float64)
    for c in range(C):
        tc = t_cluster[c] if np.isfinite(t_cluster[c]) else global_t
        nc = n_cluster[c]
        wc = nc / (nc + shrink_tau) if (nc + shrink_tau) > 0 else 0.0
        t_mix_cluster[c] = wc * tc + (1.0 - wc) * global_t
    
    null_id = C - 1
    t_mix_cluster[null_id] = global_t # null cluster always uses global threshold


    # ------------------------------------------------------------
    # per-class SCCP threshold
    # ------------------------------------------------------------
    t_sccp = np.zeros(K, dtype=np.float64)
    for k in range(K):
        ck = int(c_labels[k])
        if ck < 0 or ck >= C:
            raise ValueError(f"class {k}: cluster id {ck} out of range [0, {C-1}]")
        t_cluster = np.full(C, np.nan, dtype=np.float64)
        for c in range(C):
            cls_in_c = np.where(c_labels == c)[0]
            m = np.isin(y_cal, cls_in_c)
            if m.any():
                t_cluster[c] = quantile_upper(scores_cal[m], alpha)
            else:
                t_cluster[c] = global_t

        counts_cal = np.bincount(y_cal, minlength=K)
        t_sccp = np.zeros(K, dtype=np.float64)
        for y  in range(K):
            ny = int(counts_cal[y])
            tau_y = shrink_tau / (shrink_tau + ny) if (shrink_tau + ny) > 0 else 1.0
            t_sccp[y] = (1.0 - tau_y) * t_cluster[c_labels[y]] + tau_y * global_t
    # ------------------------------------------------------------

    return t_class, t_cluster, global_t, t_mix_cluster, t_sccp, c_labels

# -------------------------
# SCCP-only evaluator for sweep
# -------------------------
def eval_one_method_sccp(
    P_test: np.ndarray,
    y_test: np.ndarray,
    t_sccp: np.ndarray,
    class2cluster: np.ndarray,
    K: int,
) -> Dict[str, float]:
    S = P_test >= (1.0 - t_sccp[None, :])
    e = eval_sets(S, y_test, K=K, class2cluster=class2cluster)
    return {
        "cov": float(e.get("coverage", np.nan)),
        "size": float(e.get("avg_size", np.nan)),
        "avg_clu": float(e.get("avg_cluster_cov", np.nan)),
        "worst_clu": float(e.get("worst_cluster_cov", np.nan)),
        "std_clu": float(e.get("std_cluster_cov", np.nan)),
    }



# ----------------------------
# Main runner
# ----------------------------

# -------------------------
# Helpers
# -------------------------
def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]

@dataclass
class Results:
    method: str
    coverage: float = float("nan")
    avg_size: float = float("nan")
    avg_class_cov: float = float("nan")
    worst_class_cov: float = float("nan")
    std_class_cov: float = float("nan")
    avg_cluster_cov: float = float("nan")
    worst_cluster_cov: float = float("nan")
    std_cluster_cov: float = float("nan")


def pick_summary(e: Dict) -> Dict:
    keys = [
        "coverage", "avg_size",
        "avg_class_cov", "worst_class_cov", "std_class_cov",
        "avg_cluster_cov", "worst_cluster_cov", "std_cluster_cov",
    ]
    return {k: e.get(k, float("nan")) for k in keys}

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--clusters", type=int, default=10)
    ap.add_argument("--tau", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="")

    # --- Top-M diagnostics ---
    ap.add_argument("--report_topM", action="store_true",
                    help="Print true-label rank / top-M accuracy diagnostics.")
    ap.add_argument("--topM_list", type=str, default="1,5,10,20,50,100,200,500,1000",
                    help="Comma-separated M values for top-M accuracy.")
    ap.add_argument("--topM_split", type=str, default="test",
                    choices=["sel", "cal", "test", "all"],
                    help="Which split(s) to report: sel/cal/test/all.")

    # --- Sweep controls ---
    ap.add_argument("--sweep", action="store_true",
                    help="Run SCCP sweep over tau/clusters and exit.")
    ap.add_argument("--tau_list", type=str, default="50",
                    help="Comma-separated tau values for sweep, e.g., 1,5,10,50,200")
    ap.add_argument("--clusters_list", type=str, default=None,
                    help="Comma-separated Kc values for sweep, e.g., 5,10,20. If None, use --clusters")

    # --- Embedding controls ---
    ap.add_argument("--embed", type=str, default="score_quantile",
                    choices=["prob_mean", "score_quantile"])
    ap.add_argument("--q_grid", type=str, default=None,
                    help="Comma-separated quantiles for score_quantile embedding, e.g., 0.5,0.6,0.7,0.8,0.9")

    args = ap.parse_args()

    print("=== DEBUG: class/cluster metrics enabled ===")
    splits = load_npz_splits(args.npz, K=args.K, fallback_split_seed=args.seed)
    # --- define tail/head classes (CIFAR-LT standard: based on train-pool counts) ---
    z_npz = np.load(args.npz, allow_pickle=True)

    if "tail_set" in z_npz.files:
        tail_set = z_npz["tail_set"]
    elif "counts_pool" in z_npz.files:
        counts_pool = np.asarray(z_npz["counts_pool"], dtype=float)
        K = args.K
        m = int(np.ceil(0.2 * K))  # 기본 20%
        order = np.argsort(counts_pool)  # ascending
        tail_set = order[:m].astype(int)
    else:
        raise ValueError("Need tail_set or counts_pool in NPZ to define tail classes.")
    
    P_cal, y_cal = splits["cal"]
    P_test, y_test = splits["test"]

    if "sel" in splits:
        P_sel, y_sel = splits["sel"]
    else:
        P_sel, y_sel = P_cal, y_cal

    q_grid = parse_float_list(args.q_grid) if args.q_grid is not None else None

    # ============================================================
    # SCCP SWEEP MODE
    # ============================================================
    if args.sweep:
        tau_vals = parse_float_list(args.tau_list)
        kc_vals = _parse_int_list(args.clusters_list) if args.clusters_list else [args.clusters]

        print(f"[file] {args.npz}")
        print(f"[alpha] {args.alpha}  [seed] {args.seed}  [embed] {args.embed}"
              f"{'' if q_grid is None else f'  [q_grid]={q_grid}'}")
        print("\nSweep results (SCCP only):")
        print("Kc | tau  |  cov  | size | covT |  szT | covH |  szH | avg_clu | worst_clu | std_clu")
        print("--------------------------------------------------------------------------------------")


        for Kc in kc_vals:
            for tau in tau_vals:
                _, _, _, _, t_sccp, class2cluster = sccp_thresholds(
                    P_sel=P_sel, y_sel=y_sel,
                    P_cal=P_cal, y_cal=y_cal,
                    alpha=args.alpha,
                    n_clusters=Kc,
                    shrink_tau=tau,
                    seed=args.seed,
                    embed_mode=args.embed,
                    q_grid=q_grid,
                )
                S = P_test >= (1.0 - t_sccp[None, :])
                res = eval_sets(S, y_test, K=args.K, class2cluster=class2cluster, tail_set=tail_set)

                cov  = res["coverage"]
                size = res["avg_size"]

                covT = res.get("cov_tail", float("nan"))
                szT  = res.get("size_tail", float("nan"))
                covH = res.get("cov_head", float("nan"))
                szH  = res.get("size_head", float("nan"))

                avg_clu   = res.get("avg_cluster_cov", float("nan"))
                worst_clu = res.get("worst_cluster_cov", float("nan"))
                std_clu   = res.get("std_cluster_cov", float("nan"))

                print(f"{Kc:2d} | {tau:4.0f} | {cov:5.3f} | {size:4.1f} | "
                    f"{covT:5.3f} | {szT:4.1f} | {covH:5.3f} | {szH:4.1f} | "
                    f"{avg_clu:7.4f} | {worst_clu:9.4f} | {std_clu:7.4f}")

        return

    # ============================================================
    # SINGLE-RUN MODE (GCP/CCCP/SCCP table)
    # ============================================================

    # --- SCCP thresholds for the single run (IMPORTANT: do not reuse sweep leftovers) ---
    t_class_s, t_cluster_s, global_t_s, t_mix_cluster_s, t_sccp_s, class2cluster_s = sccp_thresholds(
        P_sel=P_sel, y_sel=y_sel,
        P_cal=P_cal, y_cal=y_cal,
        alpha=args.alpha,
        n_clusters=args.clusters,
        shrink_tau=args.tau,
        seed=args.seed,
        embed_mode=args.embed,
        q_grid=q_grid,
        cluster_mode = 'plain',
    )
    class2cluster = class2cluster_s
    t_sccp = t_sccp_s

    # --- GCP ---
    t_global = quantile_upper(score_s1(P_cal, y_cal), args.alpha)
    S_gcp = predset_from_threshold(P_test, t_global)
    eg = eval_sets(S_gcp, y_test, K=args.K, class2cluster=class2cluster, tail_set=tail_set)

    # --- CCCP (use global fallback for missing classes) ---
    scores_cal = score_s1(P_cal, y_cal)
    K = args.K
    t_class = np.zeros(K, dtype=np.float64)
    for k in range(K):
        m = (y_cal == k)
        if m.any():
            t_class[k] = quantile_upper(scores_cal[m], args.alpha)
        else:
            t_class[k] = t_global
    S_cccp = P_test >= (1.0 - t_class[None, :])
    ec = eval_sets(S_cccp, y_test, K=args.K, class2cluster=class2cluster, tail_set=tail_set)

    # --- SCCP ---
    S_sccp = P_test >= (1.0 - t_sccp[None, :])
    es = eval_sets(S_sccp, y_test, K=args.K, class2cluster=class2cluster, tail_set=tail_set)
    rows = [
        Results("GCP", **pick_summary(eg)),
        Results("CCCP", **pick_summary(ec)),
        Results("SCCP", **pick_summary(es)),
    ]

    def _get(e, key, default=float("nan")):
        v = e.get(key, default)
        try:
            return float(v)
        except Exception:
            return default

    def _row(method: str, e: dict):
        return {
            "method": method,
            "cov": _get(e, "coverage"),
            "size": _get(e, "avg_size"),
            "covT": _get(e, "cov_tail"),
            "szT": _get(e, "size_tail"),
            "covH": _get(e, "cov_head"),
            "szH": _get(e, "size_head"),
            "avg_cls": _get(e, "avg_class_cov"),
            "worst_cls": _get(e, "worst_class_cov"),
            "std_cls": _get(e, "std_class_cov"),
            "avg_clu": _get(e, "avg_cluster_cov"),
            "worst_clu": _get(e, "worst_cluster_cov"),
            "std_clu": _get(e, "std_cluster_cov"),
        }
    tab = [
        _row("GCP", eg),
        _row("CCCP", ec),
        _row("SCCP", es),
    ]


    # --- Print results table ---
    print(f"[file] {args.npz}")
    print(f"[alpha] {args.alpha}  [clusters] {args.clusters}  [tau] {args.tau}  [seed] {args.seed}")
    print("")
    print("method |   cov  |  size |  covT |   szT |  covH |   szH | avg_cls | worst_cls | std_cls | avg_clu | worst_clu | std_clu")
    print("-" * 118)

    for r in tab:
        print(f"{r['method']:6s} | {r['cov']:6.4f} | {r['size']:5.1f} | "
            f"{r['covT']:5.3f} | {r['szT']:5.1f} | {r['covH']:5.3f} | {r['szH']:5.1f} | "
            f"{r['avg_cls']:7.4f} | {r['worst_cls']:9.4f} | {r['std_cls']:7.4f} | "
            f"{r['avg_clu']:7.4f} | {r['worst_clu']:9.4f} | {r['std_clu']:7.4f}")


    # --- Top-M diagnostics ---
    if args.report_topM:
        Ms = _parse_int_list(args.topM_list)

        def _print_topM(tag: str, P: np.ndarray, y: np.ndarray):
            s = summarize_topM(P, y, Ms=Ms)
            rq = s["rank_quantiles"]
            print(f"\n[Top-M diagnostics: {tag}] n={s['n']}")
            print(f"  rank mean={s['rank_mean']:.2f} | q50={rq['q50']:.0f} q90={rq['q90']:.0f} "
                  f"q95={rq['q95']:.0f} q99={rq['q99']:.0f}")
            print("  top-M accuracy:")
            for M in Ms:
                print(f"    top-{M:<4d}: {s['topM_acc'][M]:.4f}")

        if args.topM_split in ("sel", "all"):
            _print_topM("sel", P_sel, y_sel)
        if args.topM_split in ("cal", "all"):
            _print_topM("cal", P_cal, y_cal)
        if args.topM_split in ("test", "all"):
            _print_topM("test", P_test, y_test)

    # --- Save JSON output ---
    if args.out:
        import json
        topM_report = {}
        if args.report_topM:
            Ms = _parse_int_list(args.topM_list)
            if args.topM_split in ("sel", "all"):
                topM_report["sel"] = summarize_topM(P_sel, y_sel, Ms=Ms)
            if args.topM_split in ("cal", "all"):
                topM_report["cal"] = summarize_topM(P_cal, y_cal, Ms=Ms)
            if args.topM_split in ("test", "all"):
                topM_report["test"] = summarize_topM(P_test, y_test, Ms=Ms)

        out = {
            "npz": args.npz,
            "alpha": args.alpha,
            "clusters": args.clusters,
            "tau": args.tau,
            "seed": args.seed,
            "embed": args.embed,
            "q_grid": q_grid,
            "results": [r.__dict__ for r in rows],
            "t_global": float(t_global),
            "t_class": t_class.tolist(),
            "t_sccp": t_sccp.tolist(),
            "class2cluster": class2cluster.tolist(),
            "GCP": eg,
            "CCCP": ec,
            "SCCP": es,
            "topM": topM_report,
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()





