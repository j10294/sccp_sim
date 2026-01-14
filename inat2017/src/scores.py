import numpy as np

def nonconformity_from_probs(P, y):
    # P: (n,K), y: (n,)
    return 1.0 - P[np.arange(len(y)), y]

def conformal_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=float)
    scores = scores[~np.isnan(scores)]
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return np.partition(scores, k - 1)[k - 1]

def empirical_quantile(scores, q):
    scores = np.asarray(scores, dtype=float)
    scores = scores[~np.isnan(scores)]
    if len(scores) == 0:
        return np.inf
    return float(np.quantile(scores, q))
