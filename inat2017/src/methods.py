import numpy as np
from .scores import nonconformity_from_probs, conformal_quantile

def thresholds_cccp(P_cq, y_cq, y2c, C, alpha):
    A = nonconformity_from_probs(P_cq, y_cq)
    q_c = np.full(C, np.inf)
    for c in range(C):
        mask = (y2c[y_cq] == c)
        q_c[c] = conformal_quantile(A[mask], alpha)
    return q_c

def threshold_global(P_cq, y_cq, alpha):
    A = nonconformity_from_probs(P_cq, y_cq)
    return conformal_quantile(A, alpha)

def per_class_thresholds_from_cluster(q_c, y2c):
    K = len(y2c)
    return np.array([q_c[y2c[k]] for k in range(K)], dtype=float)

def per_class_thresholds_scc(q_c, q_g, w_c, y2c):
    K = len(y2c)
    qk = np.empty(K, dtype=float)
    for k in range(K):
        c = y2c[k]
        qk[k] = w_c[c] * q_c[c] + (1.0 - w_c[c]) * q_g
    return qk

def predict_sets(P_te, qk):
    # A(x,k)=1-Pk(x). include k if A<=qk[k]  <=>  Pk(x) >= 1-qk[k]
    A = 1.0 - P_te
    return [set(np.where(A[i] <= qk)[0].tolist()) for i in range(P_te.shape[0])]
