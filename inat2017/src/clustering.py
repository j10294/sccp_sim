import numpy as np
from sklearn.cluster import KMeans
from .scores import empirical_quantile

def build_class_quantile_vectors(A_true, y, K, alpha, qs=(0.5,0.6,0.7,0.8,0.9)):
    """
    A_true: (n,) scores A(x_i, y_i)
    y: (n,) true labels
    returns V: (K, len(qs)+1) with last entry q=1-alpha
    """
    qs = list(qs) + [1.0 - alpha]
    V = np.full((K, len(qs)), np.nan, dtype=float)
    for k in range(K):
        sk = A_true[y == k]
        for j, q in enumerate(qs):
            V[k, j] = empirical_quantile(sk, q)
    # missing class (no samples) 처리: 큰 값으로 보내거나 global median으로 대체
    col_med = np.nanmedian(V, axis=0)
    inds = np.where(np.isnan(V))
    V[inds] = np.take(col_med, inds[1])
    return V

def cluster_classes(V, n_clusters, seed=0):
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    y2c = km.fit_predict(V)  # (K,)
    return y2c
