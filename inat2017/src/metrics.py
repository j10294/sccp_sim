import numpy as np

def coverage(C_list, y):
    return float(np.mean([y[i] in C_list[i] for i in range(len(y))]))

def avg_size(C_list):
    return float(np.mean([len(s) for s in C_list]))

def cluster_coverages(C_list, y, y2c, C):
    cov_c = np.full(C, np.nan)
    for c in range(C):
        idx = np.where(y2c[y] == c)[0]
        if len(idx) == 0:
            continue
        cov_c[c] = np.mean([y[i] in C_list[i] for i in idx])
    return cov_c

