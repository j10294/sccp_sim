#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-100 simulation runner for APS / RAPS / SCC-RAPS.

What this script does (single run):
  1) Train a simple ResNet18 on CIFAR-100 (or skip training and load checkpoint).
  2) Get softmax probabilities on tune/cal/test splits.
  3) Build class clusters (optional) using class-conditional mean predicted-prob vectors on the tune split.
  4) Run conformal calibration for each method to hit marginal coverage 1-alpha.
  5) Evaluate coverage + average set size on test.

Designed to be "good enough" for a workshop talk: reproducible, debuggable, and extensible.

Example:
  python3 run_cifar100_scc_raps.py \
    --data_root ./data --out_dir ./out/cifar100_sccraps \
    --seed 1 --epochs 10 --batch_size 256 --use_gpu \
    --alpha 0.1 --clusters 10 --lambda0 0.01 --k_reg 5
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# Optional: scikit-learn for k-means
try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None


# ---------------------------
# Repro / utils
# ---------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------
# Data split
# ---------------------------
@dataclass
class Splits:
    train_idx: np.ndarray
    tune_idx: np.ndarray
    cal_idx: np.ndarray
    test_idx: np.ndarray


def make_splits(
    n_total_train: int,
    n_train: int,
    n_tune: int,
    n_cal: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the CIFAR-100 train set indices into train/tune/cal.
    """
    assert n_train + n_tune + n_cal <= n_total_train
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total_train)
    idx_train = perm[:n_train]
    idx_tune = perm[n_train : n_train + n_tune]
    idx_cal = perm[n_train + n_tune : n_train + n_tune + n_cal]
    return idx_train, idx_tune, idx_cal


# ---------------------------
# Model + training
# ---------------------------
def build_resnet18(num_classes: int = 100) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        total_correct += int((pred == y).sum().item())
        total += x.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "acc": total_correct / max(total, 1),
    }


@torch.no_grad()
def eval_acc(model, loader, device) -> Dict[str, float]:
    model.eval()
    total_correct = 0
    total = 0
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        total_correct += int((pred == y).sum().item())
        total += x.size(0)
    return {
        "loss": total_loss / max(total, 1),
        "acc": total_correct / max(total, 1),
    }


@torch.no_grad()
def predict_proba(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list = []
    y_list = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs_list.append(p)
        y_list.append(y.numpy())
    P = np.concatenate(probs_list, axis=0)
    Y = np.concatenate(y_list, axis=0)
    return P, Y


# ---------------------------
# Conformal prediction sets: APS / RAPS / SCC-RAPS
# ---------------------------
def _sorted_probs_and_ranks(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    p: (K,) probability vector.
    Returns:
      p_sorted: (K,) sorted descending
      order: (K,) class indices in sorted order, i.e. p[order[j]] = p_sorted[j]
    """
    order = np.argsort(-p)
    p_sorted = p[order]
    return p_sorted, order


def _rank_of_label(order: np.ndarray, y: int) -> int:
    """
    order: (K,) permutation (sorted order)
    returns rank L in {1,...,K} where y is at position L-1
    """
    # Inverse perm
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return int(inv[y] + 1)


def _rho_x_of_rank(p_sorted: np.ndarray, L: int) -> float:
    """
    rho_x(y): sum of probs strictly greater than pi_y.
    Under strict sorting, rho_x for label at rank L equals sum_{j< L} p_sorted[j].
    """
    if L <= 1:
        return 0.0
    return float(p_sorted[: L - 1].sum())


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Split conformal threshold:
      tau = (ceil((n+1)*(1-alpha)))-th smallest? (depending on score definition)
    Here, we use the standard:
      tau = quantile_{ceil((n+1)*(1-alpha))/n}(scores)
    for "scores where larger = worse" (i.e., include if score <= tau).
    """
    n = scores.size
    q = math.ceil((n + 1) * (1 - alpha)) / n
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(scores, q, method="higher"))


def scores_APS(P: np.ndarray, Y: np.ndarray, randomized: bool, seed: int) -> np.ndarray:
    """
    APS conformity score for true label:
      E_i = cumulative mass up to true label rank (inclusive), optionally randomized.
    Include y if cumulative_mass(y) <= tau.
    """
    rng = np.random.default_rng(seed)
    n, K = P.shape
    E = np.zeros(n, dtype=float)
    for i in range(n):
        p = P[i]
        p_sorted, order = _sorted_probs_and_ranks(p)
        y = int(Y[i])
        L = _rank_of_label(order, y)
        cum = float(p_sorted[:L].sum())
        if randomized:
            # subtract U * p_at_L to get randomized tie-breaking
            u = rng.uniform(0.0, 1.0)
            cum = cum - u * float(p_sorted[L - 1])
        E[i] = cum
    return E


def predset_APS(p: np.ndarray, tau: float, randomized: bool, rng: np.random.Generator) -> np.ndarray:
    """
    Return boolean mask (K,) for labels included by APS at threshold tau.
    """
    p_sorted, order = _sorted_probs_and_ranks(p)
    cum = np.cumsum(p_sorted)
    if randomized:
        u = rng.uniform(0.0, 1.0)
        # randomized inclusion rule: cum - u * p <= tau
        lhs = cum - u * p_sorted
    else:
        lhs = cum
    keep_sorted = lhs <= tau
    keep = np.zeros_like(p_sorted, dtype=bool)
    keep[keep_sorted] = True
    # map back to original label order
    mask = np.zeros(order.size, dtype=bool)
    mask[order] = keep
    return mask


def scores_RAPS(P: np.ndarray, Y: np.ndarray, lam: float, k_reg: int, randomized: bool, seed: int) -> np.ndarray:
    """
    RAPS conformity score for true label:
      E_i = cumulative mass up to true label + lam * (rank - k_reg)_+
            optionally randomized by subtracting U * p_at_rank.
    """
    rng = np.random.default_rng(seed)
    n, K = P.shape
    E = np.zeros(n, dtype=float)
    for i in range(n):
        p = P[i]
        p_sorted, order = _sorted_probs_and_ranks(p)
        y = int(Y[i])
        L = _rank_of_label(order, y)
        cum = float(p_sorted[:L].sum())
        pen = lam * max(0, L - k_reg)
        if randomized:
            u = rng.uniform(0.0, 1.0)
            cum = cum - u * float(p_sorted[L - 1])
        E[i] = cum + pen
    return E


def predset_RAPS(p: np.ndarray, tau: float, lam: float, k_reg: int, randomized: bool, rng: np.random.Generator) -> np.ndarray:
    """
    Include y if:
      cum_mass(rank(y)) - u*p_y + lam*(rank-k_reg)_+ <= tau
    """
    p_sorted, order = _sorted_probs_and_ranks(p)
    cum = np.cumsum(p_sorted)
    ranks = np.arange(1, order.size + 1)
    pen = lam * np.maximum(0, ranks - k_reg)
    if randomized:
        u = rng.uniform(0.0, 1.0)
        lhs = cum - u * p_sorted + pen
    else:
        lhs = cum + pen
    keep_sorted = lhs <= tau
    mask = np.zeros(order.size, dtype=bool)
    mask[order] = keep_sorted
    return mask


def build_clusters_from_tune(P_tune: np.ndarray, Y_tune: np.ndarray, C: int, seed: int) -> np.ndarray:
    """
    Cluster classes using class-conditional mean predicted prob vectors on tune:
      emb_y = mean_{i: Y_i=y} P_tune[i,:]
    Then run k-means on emb_y in R^K.
    Returns cluster assignment c[y] in {0,...,C-1}.
    """
    if KMeans is None:
        raise RuntimeError("scikit-learn not available. Install scikit-learn or set --clusters 1.")
    n, K = P_tune.shape
    emb = np.zeros((K, K), dtype=float)
    counts = np.zeros(K, dtype=int)
    for i in range(n):
        y = int(Y_tune[i])
        emb[y] += P_tune[i]
        counts[y] += 1
    # avoid division by zero (shouldn't happen if tune covers all classes, but safe)
    for y in range(K):
        if counts[y] > 0:
            emb[y] /= counts[y]
        else:
            # fallback: use one-hot prior
            emb[y, y] = 1.0

    km = KMeans(n_clusters=C, random_state=seed, n_init="auto")
    c = km.fit_predict(emb)
    return c.astype(int)


def cluster_reliability_weights(P_tune: np.ndarray, Y_tune: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Compute simple reliability weights per cluster based on accuracy on tune.
    w_c = 1 - acc_c (harder cluster -> larger weight).
    Normalize to mean 1.
    """
    n, K = P_tune.shape
    C = int(c.max()) + 1
    correct = np.zeros(C, dtype=float)
    total = np.zeros(C, dtype=float)
    yhat = P_tune.argmax(axis=1)
    for i in range(n):
        cl = int(c[int(Y_tune[i])])
        total[cl] += 1.0
        correct[cl] += 1.0 if int(yhat[i]) == int(Y_tune[i]) else 0.0
    acc = np.divide(correct, np.maximum(total, 1.0))
    w = 1.0 - acc
    # avoid zeros; keep mild shrinkage floor
    w = np.maximum(w, 0.05)
    w = w / np.mean(w)
    return w


def scores_SCC_RAPS(
    P: np.ndarray,
    Y: np.ndarray,
    c: np.ndarray,
    lam_c: np.ndarray,
    k_c: np.ndarray,
    randomized: bool,
    seed: int,
) -> np.ndarray:
    """
    SCC-RAPS conformity score for true label:
      E_i = cum_mass(rank) + lam_{c(y)} * (rank - k_{c(y)})_+  - u*p_y (optional)
    """
    rng = np.random.default_rng(seed)
    n, K = P.shape
    E = np.zeros(n, dtype=float)
    for i in range(n):
        p = P[i]
        p_sorted, order = _sorted_probs_and_ranks(p)
        y = int(Y[i])
        L = _rank_of_label(order, y)
        cl = int(c[y])
        lam = float(lam_c[cl])
        kk = int(k_c[cl])
        cum = float(p_sorted[:L].sum())
        pen = lam * max(0, L - kk)
        if randomized:
            u = rng.uniform(0.0, 1.0)
            cum = cum - u * float(p_sorted[L - 1])
        E[i] = cum + pen
    return E


def predset_SCC_RAPS(
    p: np.ndarray,
    tau: float,
    c: np.ndarray,
    lam_c: np.ndarray,
    k_c: np.ndarray,
    randomized: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    K = p.size
    p_sorted, order = _sorted_probs_and_ranks(p)
    cum = np.cumsum(p_sorted)
    ranks = np.arange(1, K + 1)

    # cluster depends on label y -> penalty depends on label at each sorted position
    cl_sorted = np.array([int(c[int(lbl)]) for lbl in order], dtype=int)
    lam_sorted = lam_c[cl_sorted]
    k_sorted = k_c[cl_sorted]
    pen = lam_sorted * np.maximum(0, ranks - k_sorted)

    if randomized:
        u = rng.uniform(0.0, 1.0)
        lhs = cum - u * p_sorted + pen
    else:
        lhs = cum + pen

    keep_sorted = lhs <= tau
    mask = np.zeros(K, dtype=bool)
    mask[order] = keep_sorted
    return mask


def evaluate_sets(P_test: np.ndarray, Y_test: np.ndarray, masks: List[np.ndarray]) -> Dict[str, float]:
    """
    masks: list of boolean masks (K,) per test sample
    """
    n = P_test.shape[0]
    covered = 0
    sizes = []
    for i in range(n):
        y = int(Y_test[i])
        m = masks[i]
        covered += 1 if m[y] else 0
        sizes.append(int(m.sum()))
    return {
        "coverage": covered / max(n, 1),
        "avg_size": float(np.mean(sizes)) if sizes else float("nan"),
        "p90_size": float(np.quantile(sizes, 0.90)) if sizes else float("nan"),
        "p99_size": float(np.quantile(sizes, 0.99)) if sizes else float("nan"),
    }


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./out/cifar100_sccraps")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--use_gpu", action="store_true")

    # training
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--ckpt", type=str, default="", help="optional path to load/save model checkpoint")
    ap.add_argument("--skip_train", action="store_true", help="skip training and require --ckpt to load")

    # splits
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--n_tune", type=int, default=5000)
    ap.add_argument("--n_cal", type=int, default=5000)

    # conformal
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--randomized", action="store_true", help="use randomized conformal sets")

    # RAPS / SCC-RAPS params
    ap.add_argument("--k_reg", type=int, default=5)
    ap.add_argument("--lambda_raps", type=float, default=0.01)

    # clustering
    ap.add_argument("--clusters", type=int, default=10, help="C (set 1 to disable clustering)")
    ap.add_argument("--lambda0", type=float, default=0.01, help="base lambda for SCC-RAPS")
    ap.add_argument("--shrinkage", action="store_true", help="lambda_c = lambda0 * w_c (reliability weights)")
    ap.add_argument("--k_same", action="store_true", help="use same k_reg for all clusters (recommended initially)")

    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"[device] {device}")

    # Transforms: standard CIFAR normalization
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    base_train = datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=tf_train)
    base_train_eval = datasets.CIFAR100(root=args.data_root, train=True, download=False, transform=tf_test)
    test_set = datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=tf_test)

    idx_train, idx_tune, idx_cal = make_splits(
        n_total_train=len(base_train),
        n_train=args.n_train,
        n_tune=args.n_tune,
        n_cal=args.n_cal,
        seed=args.seed,
    )

    dl_train = DataLoader(Subset(base_train, idx_train), batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    dl_tune = DataLoader(Subset(base_train_eval, idx_tune), batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
    dl_cal = DataLoader(Subset(base_train_eval, idx_cal), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    dl_test = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    # Build / load model
    model = build_resnet18(num_classes=100).to(device)

    if args.skip_train:
        assert args.ckpt, "When --skip_train is used, please provide --ckpt to load."
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"[ckpt] loaded from {args.ckpt}")
    else:
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        # simple cosine schedule
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

        for ep in range(1, args.epochs + 1):
            tr = train_one_epoch(model, dl_train, optim, device)
            va = eval_acc(model, dl_tune, device)
            scheduler.step()
            print(f"[epoch {ep:02d}] train loss={tr['loss']:.4f} acc={tr['acc']:.4f} | tune acc={va['acc']:.4f}")

        if args.ckpt:
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.ckpt)
            print(f"[ckpt] saved to {args.ckpt}")

    # Probabilities on tune / cal / test
    P_tune, Y_tune = predict_proba(model, dl_tune, device)
    P_cal, Y_cal = predict_proba(model, dl_cal, device)
    P_test, Y_test = predict_proba(model, dl_test, device)
    print(f"[probs] tune={P_tune.shape} cal={P_cal.shape} test={P_test.shape}")

    # ---------------------------
    # Build clusters for SCC-RAPS
    # ---------------------------
    if args.clusters <= 1:
        c = np.zeros(100, dtype=int)
        C = 1
        print("[cluster] disabled (C=1)")
    else:
        if KMeans is None:
            raise RuntimeError("scikit-learn not installed. Install it or set --clusters 1.")
        c = build_clusters_from_tune(P_tune, Y_tune, C=args.clusters, seed=args.seed)
        C = args.clusters
        print(f"[cluster] built: C={C}, counts={np.bincount(c)}")

    # Cluster parameters
    if args.k_same:
        k_c = np.full(C, args.k_reg, dtype=int)
    else:
        # If you want cluster-dependent k, start with same and later tune
        k_c = np.full(C, args.k_reg, dtype=int)

    if args.shrinkage and C > 1:
        w = cluster_reliability_weights(P_tune, Y_tune, c)
        lam_c = args.lambda0 * w
        print(f"[scc] shrinkage on: lambda0={args.lambda0}, w(min/mean/max)=({w.min():.3f},{w.mean():.3f},{w.max():.3f})")
    else:
        lam_c = np.full(C, args.lambda0, dtype=float)

    # ---------------------------
    # Calibrate thresholds
    # ---------------------------
    # APS
    E_aps = scores_APS(P_cal, Y_cal, randomized=args.randomized, seed=args.seed + 10)
    tau_aps = conformal_quantile(E_aps, alpha=args.alpha)

    # RAPS
    E_raps = scores_RAPS(P_cal, Y_cal, lam=args.lambda_raps, k_reg=args.k_reg, randomized=args.randomized, seed=args.seed + 20)
    tau_raps = conformal_quantile(E_raps, alpha=args.alpha)

    # SCC-RAPS
    E_scc = scores_SCC_RAPS(P_cal, Y_cal, c=c, lam_c=lam_c, k_c=k_c, randomized=args.randomized, seed=args.seed + 30)
    tau_scc = conformal_quantile(E_scc, alpha=args.alpha)

    print(f"[tau] APS={tau_aps:.6f} | RAPS={tau_raps:.6f} | SCC-RAPS={tau_scc:.6f}")

    # ---------------------------
    # Build prediction sets on test
    # ---------------------------
    rng = np.random.default_rng(args.seed + 999)
    masks_aps = []
    masks_raps = []
    masks_scc = []
    for i in range(P_test.shape[0]):
        p = P_test[i]
        masks_aps.append(predset_APS(p, tau_aps, randomized=args.randomized, rng=rng))
        masks_raps.append(predset_RAPS(p, tau_raps, lam=args.lambda_raps, k_reg=args.k_reg, randomized=args.randomized, rng=rng))
        masks_scc.append(predset_SCC_RAPS(p, tau_scc, c=c, lam_c=lam_c, k_c=k_c, randomized=args.randomized, rng=rng))

    res = {
        "args": vars(args),
        "tau": {"APS": tau_aps, "RAPS": tau_raps, "SCC_RAPS": tau_scc},
        "RAPS_params": {"lambda": args.lambda_raps, "k_reg": args.k_reg},
        "SCC_params": {
            "C": int(C),
            "lambda0": float(args.lambda0),
            "lambda_c": lam_c.tolist(),
            "k_c": k_c.tolist(),
        },
        "test": {
            "APS": evaluate_sets(P_test, Y_test, masks_aps),
            "RAPS": evaluate_sets(P_test, Y_test, masks_raps),
            "SCC_RAPS": evaluate_sets(P_test, Y_test, masks_scc),
        }
    }

    print(json.dumps(res["test"], indent=2))

    out_path = os.path.join(args.out_dir, f"cifar100_seed{args.seed:03d}.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
