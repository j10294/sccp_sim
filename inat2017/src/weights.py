import numpy as np
from .scores import nonconformity_from_probs, conformal_quantile

def learn_weights_nested(P_cw, y_cw, y2c, C, alpha, grid=None, seed=0):
    """
    nested split:
      cw1 -> compute (q_c, q_g)
      cw2 -> choose w_c by minimizing |emp_miscoverage - alpha| within each cluster
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 51)

    n = len(y_cw)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    mid = n // 2
    i1, i2 = idx[:mid], idx[mid:]

    # cw1 quantiles
    A1 = nonconformity_from_probs(P_cw[i1], y_cw[i1])
    q_g = conformal_quantile(A1, alpha)

    q_c = np.full(C, np.inf)
    for c in range(C):
        mask = (y2c[y_cw[i1]] == c)
        q_c[c] = conformal_quantile(A1[mask], alpha)

    # cw2 choose weights per cluster
    A2_mat = 1.0 - P_cw[i2]  # (n2, K)
    y2 = y_cw[i2]

    w = np.zeros(C, dtype=float)
    for c in range(C):
        idx_c = np.where(y2c[y2] == c)[0]
        if len(idx_c) == 0:
            w[c] = 0.0
            continue

        # For each sample i in cluster c, true label is y2[i]
        # prediction set membership condition for SCC: A(x, y) <= w q_c + (1-w) q_g
        # For coverage matching, only need true-label scores.
        true_scores = 1.0 - P_cw[i2][np.arange(len(y2)), y2]
        true_scores_c = true_scores[idx_c]

        best_w, best_err = 0.0, 1e9
        for ww in grid:
            thr = ww * q_c[c] + (1.0 - ww) * q_g
            miscov = np.mean(true_scores_c > thr)  # indicator Y not in set (at least for true label test)
            err = abs(miscov - alpha)
            if err < best_err:
                best_err = err
                best_w = ww
        w[c] = best_w
    return w
