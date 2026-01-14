import numpy as np
from .scores import nonconformity_from_probs
from .clustering import build_class_quantile_vectors, cluster_classes
from .weights import learn_weights_nested
from .methods import (
    thresholds_cccp, threshold_global,
    per_class_thresholds_from_cluster, per_class_thresholds_scc,
    predict_sets
)
from .metrics import coverage, avg_size, cluster_coverages

def run_one_seed(P_tr, y_tr, P_cw, y_cw, P_cq, y_cq, P_te, y_te,
                 alpha=0.1, n_clusters=50, seed=0):

    K = P_tr.shape[1]

    # 1) clustering via class quantile vectors (from training split)
    A_tr = nonconformity_from_probs(P_tr, y_tr)
    V = build_class_quantile_vectors(A_tr, y_tr, K=K, alpha=alpha)
    y2c = cluster_classes(V, n_clusters=n_clusters, seed=seed)
    C = n_clusters

    # 2) learn weights (nested split on cw)
    w_c = learn_weights_nested(P_cw, y_cw, y2c, C, alpha, seed=seed)

    # 3) estimate thresholds on cq
    q_c = thresholds_cccp(P_cq, y_cq, y2c, C, alpha)
    q_g = threshold_global(P_cq, y_cq, alpha)

    qk_cccp = per_class_thresholds_from_cluster(q_c, y2c)
    qk_scc  = per_class_thresholds_scc(q_c, q_g, w_c, y2c)
    qk_glob = np.full(K, q_g, dtype=float)

    # 4) predict sets
    C_cccp = predict_sets(P_te, qk_cccp)
    C_scc  = predict_sets(P_te, qk_scc)
    C_glob = predict_sets(P_te, qk_glob)

    # 5) metrics
    out = {}
    for name, Cset in [("cccp", C_cccp), ("scc", C_scc), ("glob", C_glob)]:
        out[f"cov_{name}"] = coverage(Cset, y_te)
        out[f"size_{name}"] = avg_size(Cset)
        cov_c = cluster_coverages(Cset, y_te, y2c, C)
        out[f"worst_cluster_cov_{name}"] = float(np.nanmin(cov_c))
        out[f"var_cluster_cov_{name}"] = float(np.nanvar(cov_c))
    return out
