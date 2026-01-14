import os
import json
import argparse
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset


# -------------------------
# Reproducibility utilities
# -------------------------
def set_seed(seed: int, deterministic: bool = True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ----------------------------------------
# Imbalance construction (train/test both)
# ----------------------------------------
def make_mild_imbalance_indices(
    targets,
    tail_frac: float = 0.2,
    tail_classes: int = 30,
    seed: int = 0,
    min_keep_per_class: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      idx_keep: indices kept after downsampling tail classes
      tail_classes_arr: array of tail class labels used
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(targets, dtype=int)
    K = int(y.max() + 1)

    tail_classes = int(tail_classes)
    tail_classes = max(0, min(tail_classes, K))
    tail_set = np.array(list(range(K - tail_classes, K)), dtype=int)

    idx_keep = []
    tail_set_s = set(tail_set.tolist())

    for k in range(K):
        idx_k = np.where(y == k)[0]
        rng.shuffle(idx_k)

        if k in tail_set_s:
            m = max(min_keep_per_class, int(len(idx_k) * float(tail_frac)))
        else:
            m = len(idx_k)

        idx_keep.append(idx_k[:m])

    idx_keep = np.concatenate(idx_keep).astype(int)
    rng.shuffle(idx_keep)
    return idx_keep, tail_set


def class_counts_from_indices(targets, idx: np.ndarray, K: int) -> np.ndarray:
    y = np.asarray(targets, dtype=int)
    yy = y[idx]
    return np.bincount(yy, minlength=K)


# --------------------------------------
# Stratified split within an index pool
# --------------------------------------
def stratified_split_from_pool(
    pool_idx: np.ndarray,
    targets,
    n_train: int,
    n_select: int,
    n_calib: int,
    seed: int,
    # Ensure calibration has at least this many examples per class when possible
    min_calib_per_class: int = 5,
    # Optionally ensure train has minimum per class too (can be 0)
    min_train_per_class: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    pool_idx: indices into the base dataset (train set) after imbalance.
    We stratify by class within this pool.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(targets, dtype=int)
    K = int(y.max() + 1)

    pool_idx = np.asarray(pool_idx, dtype=int)
    # group indices by class
    by_class = [pool_idx[y[pool_idx] == k] for k in range(K)]
    for k in range(K):
        rng.shuffle(by_class[k])

    # helper: allocate counts per class proportional to availability
    def allocate_counts(total: int, avail: np.ndarray) -> np.ndarray:
        if total <= 0:
            return np.zeros_like(avail)
        if avail.sum() == 0:
            return np.zeros_like(avail)
        w = avail / avail.sum()
        base = np.floor(w * total).astype(int)
        # distribute remainder by largest fractional parts
        rem = total - base.sum()
        if rem > 0:
            frac = (w * total) - base
            order = np.argsort(-frac)
            for i in order[:rem]:
                base[i] += 1
        # clip to availability
        base = np.minimum(base, avail)
        # if clipping reduced sum, greedily fill where possible
        short = total - base.sum()
        if short > 0:
            room = avail - base
            order = np.argsort(-room)
            for i in order:
                if short <= 0:
                    break
                add = min(short, room[i])
                base[i] += add
                short -= add
        return base

    avail = np.array([len(by_class[k]) for k in range(K)], dtype=int)

    # First, reserve minimums if feasible
    cal_min = np.minimum(avail, min_calib_per_class)
    # But if sum(cal_min) > n_calib, reduce minimums (rare but possible)
    if cal_min.sum() > n_calib:
        # scale down minimums uniformly, then fix remainder
        cal_min = allocate_counts(n_calib, avail)

    avail_after_calmin = avail - cal_min

    tr_min = np.minimum(avail_after_calmin, min_train_per_class)
    if tr_min.sum() > n_train:
        tr_min = allocate_counts(n_train, avail_after_calmin)

    avail_after_mins = avail - cal_min - tr_min

    # allocate remaining counts proportionally
    n_cal_rem = n_calib - cal_min.sum()
    n_tr_rem = n_train - tr_min.sum()
    n_sel_total = n_select

    cal_add = allocate_counts(n_cal_rem, avail_after_mins)
    avail_after_cal = avail_after_mins - cal_add

    tr_add = allocate_counts(n_tr_rem, avail_after_cal)
    avail_after_tr = avail_after_cal - tr_add

    sel_add = allocate_counts(n_sel_total, avail_after_tr)
    # Any leftover (because not enough samples) will be handled by final check.

    # Now actually pick indices
    idx_cal = []
    idx_tr = []
    idx_sel = []

    for k in range(K):
        n_c = int(cal_min[k] + cal_add[k])
        n_t = int(tr_min[k] + tr_add[k])
        n_s = int(sel_add[k])

        arr = by_class[k]
        # order: calib first, then train, then select
        idx_cal.append(arr[:n_c])
        idx_tr.append(arr[n_c:n_c + n_t])
        idx_sel.append(arr[n_c + n_t:n_c + n_t + n_s])

    idx_cal = np.concatenate(idx_cal) if len(idx_cal) else np.array([], dtype=int)
    idx_tr = np.concatenate(idx_tr) if len(idx_tr) else np.array([], dtype=int)
    idx_sel = np.concatenate(idx_sel) if len(idx_sel) else np.array([], dtype=int)

    # Final sanity: ensure disjoint and within pool
    # (Disjointness guaranteed by slicing, but check sizes)
    # If pool is too small, these may be smaller than requested -> error to keep honest.
    if len(idx_tr) != n_train or len(idx_sel) != n_select or len(idx_cal) != n_calib:
        raise ValueError(
            f"Not enough samples in imbalanced pool to satisfy split sizes.\n"
            f"Requested (tr/sel/cal)=({n_train}/{n_select}/{n_calib}), "
            f"got ({len(idx_tr)}/{len(idx_sel)}/{len(idx_cal)}).\n"
            f"Try reducing split sizes or using fraction-based split."
        )

    # Shuffle within each split (optional)
    rng.shuffle(idx_tr)
    rng.shuffle(idx_sel)
    rng.shuffle(idx_cal)
    return idx_tr, idx_sel, idx_cal


# -------------------------
# Model + probability export
# -------------------------
@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    probs_list = []
    y_list = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probs_list.append(p)
        y_list.append(y.numpy())
    P = np.concatenate(probs_list, axis=0)
    Y = np.concatenate(y_list, axis=0)
    return P, Y


@dataclass
class RunMeta:
    seed_data: int
    seed_train: int
    deterministic: bool
    tail_frac: float
    tail_classes: int
    min_keep_per_class: int
    split_mode: str
    n_train: int
    n_select: int
    n_calib: int
    min_calib_per_class: int
    min_train_per_class: int
    epochs: int
    batch_size: int
    lr: float


def main():
    parser = argparse.ArgumentParser()

    # Imbalance
    parser.add_argument("--imbalance", action="store_true", help="apply mild class imbalance")
    parser.add_argument("--tail_frac", type=float, default=0.2)
    parser.add_argument("--tail_classes", type=int, default=30)
    parser.add_argument("--min_keep_per_class", type=int, default=1)

    # Split control
    parser.add_argument("--split_mode", type=str, default="counts", choices=["counts", "fracs"])
    parser.add_argument("--n_train", type=int, default=24000)
    parser.add_argument("--n_select", type=int, default=8000)
    parser.add_argument("--n_calib", type=int, default=8000)

    parser.add_argument("--train_frac", type=float, default=0.6)
    parser.add_argument("--select_frac", type=float, default=0.2)
    parser.add_argument("--calib_frac", type=float, default=0.2)

    parser.add_argument("--min_calib_per_class", type=int, default=5)
    parser.add_argument("--min_train_per_class", type=int, default=0)

    # Paths
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./out/cifar100_probs_imbworld")

    # Seeds
    parser.add_argument("--seed_data", type=int, default=1, help="seed for imbalance + splitting")
    parser.add_argument("--seed_train", type=int, default=1, help="seed for model init + training order")
    parser.add_argument("--deterministic", action="store_true", help="enable cudnn deterministic mode")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_gpu", action="store_true")

    args = parser.parse_args()

    # Seeds: separate for data pipeline vs training pipeline
    set_seed(args.seed_data, deterministic=args.deterministic)

    device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
    print(f"[device] {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # CIFAR-100 normalization commonly used
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # Load datasets
    train_base_aug = torchvision.datasets.CIFAR100(
        root=args.data_root, train=True, download=True, transform=train_tf
    )
    train_base_eval = torchvision.datasets.CIFAR100(
        root=args.data_root, train=True, download=False, transform=eval_tf
    )
    test_base_eval = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True, transform=eval_tf
    )

    K = 100

    # 1) Create imbalanced "world" indices for train and test (same rule, same seed_data)
    if args.imbalance:
        idx_train_pool, tail_set = make_mild_imbalance_indices(
            train_base_eval.targets,
            tail_frac=args.tail_frac,
            tail_classes=args.tail_classes,
            seed=args.seed_data,
            min_keep_per_class=args.min_keep_per_class,
        )
        # For test, we use the SAME tail_set definition (last tail_classes),
        # and apply the same tail_frac via the same function (it uses class labels).
        idx_test_imb, _ = make_mild_imbalance_indices(
            test_base_eval.targets,
            tail_frac=args.tail_frac,
            tail_classes=args.tail_classes,
            seed=args.seed_data,
            min_keep_per_class=args.min_keep_per_class,
        )
    else:
        idx_train_pool = np.arange(len(train_base_eval), dtype=int)
        idx_test_imb = np.arange(len(test_base_eval), dtype=int)
        tail_set = np.array([], dtype=int)

    print("[debug] train_base size =", len(train_base_eval))
    print("[debug] test_base  size =", len(test_base_eval))
    print("[debug] len(idx_train_pool_imb) =", len(idx_train_pool))
    print("[debug] len(idx_test_imb)       =", len(idx_test_imb))
    if args.imbalance:
        print("[debug] tail classes =", tail_set.tolist())

    # 2) Decide split sizes
    if args.split_mode == "fracs":
        # compute from imbalanced train pool size
        N = len(idx_train_pool)
        # normalize fractions
        fr = np.array([args.train_frac, args.select_frac, args.calib_frac], dtype=float)
        if np.any(fr < 0) or fr.sum() <= 0:
            raise ValueError("Fractions must be nonnegative and sum to a positive value.")
        fr = fr / fr.sum()
        n_train = int(np.floor(fr[0] * N))
        n_select = int(np.floor(fr[1] * N))
        n_calib = int(np.floor(fr[2] * N))
        # fix remainder to calibration (or train)
        rem = N - (n_train + n_select + n_calib)
        n_calib += rem
    else:
        n_train = int(args.n_train)
        n_select = int(args.n_select)
        n_calib = int(args.n_calib)

    # sanity
    if n_train + n_select + n_calib > len(idx_train_pool):
        raise ValueError(
            f"Requested split sizes exceed imbalanced train pool.\n"
            f"Requested sum={n_train+n_select+n_calib}, pool={len(idx_train_pool)}.\n"
            f"Use --split_mode fracs or reduce n_*."
        )

    print(f"[split sizes] n_train={n_train}, n_select={n_select}, n_calib={n_calib}")

    # 3) Stratified split from imbalanced train pool
    idx_tr, idx_sel, idx_cal = stratified_split_from_pool(
        pool_idx=idx_train_pool,
        targets=train_base_eval.targets,
        n_train=n_train,
        n_select=n_select,
        n_calib=n_calib,
        seed=args.seed_data,
        min_calib_per_class=args.min_calib_per_class,
        min_train_per_class=args.min_train_per_class,
    )

    # 4) Build datasets
    ds_tr = Subset(train_base_aug, idx_tr)        # augmentation for training
    ds_sel = Subset(train_base_eval, idx_sel)     # eval transform
    ds_cal = Subset(train_base_eval, idx_cal)     # eval transform
    ds_te = Subset(test_base_eval, idx_test_imb)  # imbalanced test world

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    dl_sel = DataLoader(ds_sel, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    dl_cal = DataLoader(ds_cal, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # 5) Model
    # training seed separately
    set_seed(args.seed_train, deterministic=args.deterministic)

    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # For short epochs, keep scheduler simple
    if args.epochs >= 10:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[max(args.epochs // 2, 1)], gamma=0.1
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 2, 1), gamma=0.1)

    # 6) Train
    model.train()
    for ep in range(1, args.epochs + 1):
        total, correct, running_loss = 0, 0, 0.0
        for x, y in dl_tr:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            total += x.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()

        scheduler.step()
        print(f"[epoch {ep}] loss={running_loss/total:.4f} acc={correct/total:.4f}")

    # 7) Predict probabilities
    p_sel, y_sel = predict_proba(model, dl_sel, device)
    p_cal, y_cal = predict_proba(model, dl_cal, device)
    p_tst, y_tst = predict_proba(model, dl_te, device)

    # 8) Save metadata (histograms etc.)
    counts_pool = class_counts_from_indices(train_base_eval.targets, idx_train_pool, K)
    counts_tr = class_counts_from_indices(train_base_eval.targets, idx_tr, K)
    counts_sel = class_counts_from_indices(train_base_eval.targets, idx_sel, K)
    counts_cal = class_counts_from_indices(train_base_eval.targets, idx_cal, K)
    counts_test = class_counts_from_indices(test_base_eval.targets, idx_test_imb, K)

    meta = RunMeta(
        seed_data=args.seed_data,
        seed_train=args.seed_train,
        deterministic=bool(args.deterministic),
        tail_frac=float(args.tail_frac),
        tail_classes=int(args.tail_classes),
        min_keep_per_class=int(args.min_keep_per_class),
        split_mode=args.split_mode,
        n_train=int(n_train),
        n_select=int(n_select),
        n_calib=int(n_calib),
        min_calib_per_class=int(args.min_calib_per_class),
        min_train_per_class=int(args.min_train_per_class),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
    )

    tag = f"seedD{args.seed_data}_seedT{args.seed_train}_e{args.epochs}_bs{args.batch_size}"
    if args.imbalance:
        tag += f"_imb_tail{args.tail_classes}_frac{args.tail_frac}"

    out_path = os.path.join(args.out_dir, f"cifar100_imbworld_{tag}.npz")
    np.savez_compressed(
        out_path,
        p_sel=p_sel, y_sel=y_sel,
        p_cal=p_cal, y_cal=y_cal,
        p_tst=p_tst, y_tst=y_tst,
        idx_tr=idx_tr, idx_sel=idx_sel, idx_cal=idx_cal,
        idx_train_pool=idx_train_pool,
        idx_test_imb=idx_test_imb,
        tail_set=tail_set,
        counts_pool=counts_pool,
        counts_tr=counts_tr,
        counts_sel=counts_sel,
        counts_cal=counts_cal,
        counts_test=counts_test,
        meta_json=np.array([json.dumps(asdict(meta))], dtype=object),
    )

    print(f"[saved] {out_path}")
    print("[shapes] p_sel", p_sel.shape, "p_cal", p_cal.shape, "p_tst", p_tst.shape)
    print("[counts] train_pool sum", counts_pool.sum(),
          "train", counts_tr.sum(), "sel", counts_sel.sum(), "cal", counts_cal.sum(),
          "test_imb", counts_test.sum())


if __name__ == "__main__":
    main()
