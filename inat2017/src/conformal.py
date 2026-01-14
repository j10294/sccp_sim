# src/conformal.py
import numpy as np

def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile for scores (nonconformity)."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[~np.isnan(scores)]
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return np.partition(scores, k - 1)[k - 1]
