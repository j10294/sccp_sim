import os
import json
import argparse
from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights,
    ResNeXt50_32X4D_Weights, ResNeXt101_32X8D_Weights
)



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


# -----------------------------------------
# CIFAR-LT construction (train indices only)
# -----------------------------------------
def _get_class_indices_from_targets(targets, K: int):
    y = np.asarray(targets, dtype=int)
    cls_idx = [np.where(y == k)[0] for k in range(K)]
    return cls_idx


def _lt_counts_exp(img_max: int, K: int, imb_factor: float) -> np.ndarray:
    # Standard: n_k = img_max * (1/IF)^(k/(K-1)), with k=0..K-1
    # Class order 0..K-1 (deterministic). Many papers use this convention.
    counts = []
    for k in range(K):
        n_k = img_max * (1.0 / float(imb_factor)) ** (k / (K - 1))
        counts.append(int(np.round(n_k)))
    counts = np.array(counts, dtype=int)
    counts = np.maximum(counts, 1)  # keep at least 1 per class
    return counts


def _lt_counts_step(img_max: int, K: int, imb_factor: float) -> np.ndarray:
    # Step imbalance: first half img_max, second half img_max/IF
    counts = np.full(K, img_max, dtype=int)
    counts[K // 2 :] = int(np.round(img_max / float(imb_factor)))
    counts = np.maximum(counts, 1)
    return counts


def make_cifar_lt_indices(
    targets,
    K: int,
    imb_type: str = "exp",
    imb_factor: float = 100.0,
    seed: int = 1,
    img_max_per_class: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build CIFAR-LT indices from a balanced base dataset by class-wise subsampling.

    Returns:
      idx_keep: selected indices (shuffled)
      cls_counts: selected counts per class (length K)
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(targets, dtype=int)

    cls_idx = _get_class_indices_from_targets(y, K)
    base_counts = np.array([len(cls_idx[k]) for k in range(K)], dtype=int)

    if img_max_per_class is None:
        img_max = int(base_counts.min())  # CIFAR train: typically 500
    else:
        img_max = int(img_max_per_class)

    if imb_type == "exp":
        cls_counts = _lt_counts_exp(img_max, K, imb_factor)
    elif imb_type == "step":
        cls_counts = _lt_counts_step(img_max, K, imb_factor)
    else:
        raise ValueError(f"Unknown imb_type={imb_type}. Use exp or step.")

    # clip to availability
    cls_counts = np.minimum(cls_counts, base_counts)
    cls_counts = np.maximum(cls_counts, 1)

    idx_keep = []
    for k in range(K):
        idx_k = np.array(cls_idx[k], dtype=int)
        rng.shuffle(idx_k)
        idx_keep.append(idx_k[: cls_counts[k]])
    idx_keep = np.concatenate(idx_keep).astype(int)
    rng.shuffle(idx_keep)
    return idx_keep, cls_counts


def class_counts_from_indices(targets, idx: np.ndarray, K: int) -> np.ndarray:
    y = np.asarray(targets, dtype=int)
    yy = y[np.asarray(idx, dtype=int)]
    return np.bincount(yy, minlength=K)


def tail_set_from_counts(cls_counts: np.ndarray, tail_frac: float) -> np.ndarray:
    """
    Define tail classes as the smallest-count classes.
    tail_frac=0.2 -> bottom 20% classes by count.
    """
    K = len(cls_counts)
    m = int(np.ceil(float(tail_frac) * K))
    m = max(0, min(m, K))
    order = np.argsort(cls_counts)  # ascending
    return order[:m].astype(int)

# --------------------------------
# Model 
# -------------------------------
def build_model(arch: str, num_classes: int = 100, pretrained: bool = True):
    # 1) Load ImageNet-pretrained weights (for the body)
    if arch == "resnet18":
        w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet18(weights=w)
    elif arch == "resnet34":
        w = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet34(weights=w)
    elif arch == "resnet50":
        w = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet50(weights=w)
    elif arch == "resnet101":
        w = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet101(weights=w)
    elif arch == "resnet152":
        w = ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet152(weights=w)
    elif arch == "resnext50_32x4d":
        w = ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnext50_32x4d(weights=w)
    elif arch == "resnext101_32x8d":
        w = ResNeXt101_32X8D_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnext101_32x8d(weights=w)
    else:
        raise ValueError(arch)

    # 2) CIFAR stem (keeps 32x32 resolution)
    #    NOTE: conv1/maxpool are now randomly initialized due to replacement.
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()

    # 3) Replace classifier head for CIFAR-100
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

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
    min_calib_per_class: int = 5,
    min_train_per_class: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.asarray(targets, dtype=int)
    K = int(y.max() + 1)

    pool_idx = np.asarray(pool_idx, dtype=int)
    by_class = [pool_idx[y[pool_idx] == k] for k in range(K)]
    for k in range(K):
        rng.shuffle(by_class[k])

    def allocate_counts(total: int, avail: np.ndarray) -> np.ndarray:
        if total <= 0 or avail.sum() == 0:
            return np.zeros_like(avail)
        w = avail / avail.sum()
        base = np.floor(w * total).astype(int)
        rem = total - base.sum()
        if rem > 0:
            frac = (w * total) - base
            order = np.argsort(-frac)
            for i in order[:rem]:
                base[i] += 1
        base = np.minimum(base, avail)
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

    cal_min = np.minimum(avail, min_calib_per_class)
    if cal_min.sum() > n_calib:
        cal_min = allocate_counts(n_calib, avail)
    avail_after_calmin = avail - cal_min

    tr_min = np.minimum(avail_after_calmin, min_train_per_class)
    if tr_min.sum() > n_train:
        tr_min = allocate_counts(n_train, avail_after_calmin)
    avail_after_mins = avail - cal_min - tr_min

    n_cal_rem = n_calib - cal_min.sum()
    n_tr_rem = n_train - tr_min.sum()

    cal_add = allocate_counts(n_cal_rem, avail_after_mins)
    avail_after_cal = avail_after_mins - cal_add

    tr_add = allocate_counts(n_tr_rem, avail_after_cal)
    avail_after_tr = avail_after_cal - tr_add

    sel_add = allocate_counts(n_select, avail_after_tr)

    idx_cal, idx_tr, idx_sel = [], [], []
    for k in range(K):
        n_c = int(cal_min[k] + cal_add[k])
        n_t = int(tr_min[k] + tr_add[k])
        n_s = int(sel_add[k])

        arr = by_class[k]
        idx_cal.append(arr[:n_c])
        idx_tr.append(arr[n_c:n_c + n_t])
        idx_sel.append(arr[n_c + n_t:n_c + n_t + n_s])

    idx_cal = np.concatenate(idx_cal) if len(idx_cal) else np.array([], dtype=int)
    idx_tr = np.concatenate(idx_tr) if len(idx_tr) else np.array([], dtype=int)
    idx_sel = np.concatenate(idx_sel) if len(idx_sel) else np.array([], dtype=int)

    if len(idx_tr) != n_train or len(idx_sel) != n_select or len(idx_cal) != n_calib:
        raise ValueError(
            f"Not enough samples in LT pool to satisfy split sizes.\n"
            f"Requested (tr/sel/cal)=({n_train}/{n_select}/{n_calib}), "
            f"got ({len(idx_tr)}/{len(idx_sel)}/{len(idx_cal)}).\n"
            f"Reduce n_* or use --split_mode fracs."
        )

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

    # LT params
    lt: bool
    imb_type: str
    imb_factor: float
    img_max_per_class: int | None
    tail_frac: float

    # Split params
    split_mode: str
    n_train: int
    n_select: int
    n_calib: int
    min_calib_per_class: int
    min_train_per_class: int

    # Train params
    epochs: int
    batch_size: int
    lr: float

    # Test mode
    lt_test: bool


def main():
    parser = argparse.ArgumentParser()

    # Long-tail (CIFAR-LT)
    parser.add_argument("--lt", action="store_true", help="use CIFAR-LT on train split (standard)")
    parser.add_argument("--imb_type", type=str, default="exp", choices=["exp", "step"])
    parser.add_argument("--imb_factor", type=float, default=100.0, help="Imbalance factor IF (e.g., 10/50/100)")
    parser.add_argument("--img_max_per_class", type=int, default=None, help="override max images per class (default=min count)")
    parser.add_argument("--tail_frac", type=float, default=0.2, help="define tail classes as bottom tail_frac by count")
    parser.add_argument("--lt_test", action="store_true", help="ALTERNATIVE (non-standard): apply same LT rule to test too")

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
    parser.add_argument("--out_dir", type=str, default="./out/cifar100_probs_cifarLT")

    # Seeds
    parser.add_argument("--seed_data", type=int, default=1, help="seed for LT+splitting")
    parser.add_argument("--seed_train", type=int, default=1, help="seed for init+training")
    parser.add_argument("--deterministic", action="store_true")

    # Training
    parser.add_argument("--arch",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d"],
    )

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_gpu", action="store_true")

    args = parser.parse_args()

    # Data seed (LT + split)
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

    # 1) Build LT pool on train
    if args.lt:
        idx_train_pool, cls_counts_pool = make_cifar_lt_indices(
            train_base_eval.targets,
            K=K,
            imb_type=args.imb_type,
            imb_factor=args.imb_factor,
            seed=args.seed_data,
            img_max_per_class=args.img_max_per_class,
        )
    else:
        idx_train_pool = np.arange(len(train_base_eval), dtype=int)
        cls_counts_pool = class_counts_from_indices(train_base_eval.targets, idx_train_pool, K)

    tail_set = tail_set_from_counts(cls_counts_pool, args.tail_frac)

    # 2) Test indices: standard is full test
    if args.lt_test and args.lt:
        idx_test, cls_counts_test = make_cifar_lt_indices(
            test_base_eval.targets,
            K=K,
            imb_type=args.imb_type,
            imb_factor=args.imb_factor,
            seed=args.seed_data,
            img_max_per_class=None,  # test has 100 per class in CIFAR100
        )
    else:
        idx_test = np.arange(len(test_base_eval), dtype=int)
        cls_counts_test = class_counts_from_indices(test_base_eval.targets, idx_test, K)

    print("[pool] train pool size =", len(idx_train_pool))
    print("[pool] test  size      =", len(idx_test))
    if args.lt:
        print(f"[LT] imb_type={args.imb_type} IF={args.imb_factor} img_max={args.img_max_per_class}")
        print("[LT] train pool class count stats:",
              f"min={cls_counts_pool.min()}, median={int(np.median(cls_counts_pool))}, max={cls_counts_pool.max()}")
    print("[tail] tail_frac =", args.tail_frac, "tail_classes =", len(tail_set))

    # 3) Decide split sizes
    if args.split_mode == "fracs":
        N = len(idx_train_pool)
        fr = np.array([args.train_frac, args.select_frac, args.calib_frac], dtype=float)
        if np.any(fr < 0) or fr.sum() <= 0:
            raise ValueError("Fractions must be nonnegative and sum to a positive value.")
        fr = fr / fr.sum()
        n_train = int(np.floor(fr[0] * N))
        n_select = int(np.floor(fr[1] * N))
        n_calib = int(np.floor(fr[2] * N))
        rem = N - (n_train + n_select + n_calib)
        n_calib += rem
    else:
        n_train = int(args.n_train)
        n_select = int(args.n_select)
        n_calib = int(args.n_calib)

    if n_train + n_select + n_calib > len(idx_train_pool):
        raise ValueError(
            f"Requested split sizes exceed train pool.\n"
            f"Requested sum={n_train+n_select+n_calib}, pool={len(idx_train_pool)}.\n"
            f"Use --split_mode fracs or reduce n_*."
        )

    print(f"[split sizes] n_train={n_train}, n_select={n_select}, n_calib={n_calib}")

    # 4) Stratified split from LT pool
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

    # 5) Build datasets
    ds_tr = Subset(train_base_aug, idx_tr)
    ds_sel = Subset(train_base_eval, idx_sel)
    ds_cal = Subset(train_base_eval, idx_cal)
    ds_te = Subset(test_base_eval, idx_test)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    dl_sel = DataLoader(ds_sel, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    dl_cal = DataLoader(ds_cal, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # 6) Model (training seed separately)
    set_seed(args.seed_train, deterministic=args.deterministic)

    model = build_model(args.arch, num_classes=K, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # common CIFAR schedule: milestones at 100/150 for 200 epochs
    if args.epochs >= 200:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[max(args.epochs // 2, 1)], gamma=0.1
        )

    # 7) Train
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

    # 8) Predict probabilities
    p_sel, y_sel = predict_proba(model, dl_sel, device)
    p_cal, y_cal = predict_proba(model, dl_cal, device)
    p_tst, y_tst = predict_proba(model, dl_te, device)

    # 9) Counts (for reporting tail/head)
    counts_pool = class_counts_from_indices(train_base_eval.targets, idx_train_pool, K)
    counts_tr = class_counts_from_indices(train_base_eval.targets, idx_tr, K)
    counts_sel = class_counts_from_indices(train_base_eval.targets, idx_sel, K)
    counts_cal = class_counts_from_indices(train_base_eval.targets, idx_cal, K)
    counts_test = class_counts_from_indices(test_base_eval.targets, idx_test, K)

    meta = RunMeta(
        seed_data=args.seed_data,
        seed_train=args.seed_train,
        deterministic=bool(args.deterministic),

        lt=bool(args.lt),
        imb_type=str(args.imb_type),
        imb_factor=float(args.imb_factor),
        img_max_per_class=args.img_max_per_class if args.img_max_per_class is None else int(args.img_max_per_class),
        tail_frac=float(args.tail_frac),

        split_mode=str(args.split_mode),
        n_train=int(n_train),
        n_select=int(n_select),
        n_calib=int(n_calib),
        min_calib_per_class=int(args.min_calib_per_class),
        min_train_per_class=int(args.min_train_per_class),

        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),

        lt_test=bool(args.lt_test),
    )

    tag = f"{args.arch}_e{args.epochs}_bs{args.batch_size}"

    if args.lt:
        tag += f"_LT_{args.imb_type}_IF{int(args.imb_factor)}"

    if args.lt_test and args.lt:
        tag += "_LTtest"

    tag += f"_tailfrac{args.tail_frac}"
    tag += f"_cal{int(round(args.calib_frac * 100)):02d}"
    tag += f"_sd{args.seed_data}_st{args.seed_train}"



    out_path = os.path.join(args.out_dir, f"cifar100_{tag}.npz")
    np.savez_compressed(
        out_path,
        p_sel=p_sel, y_sel=y_sel,
        p_cal=p_cal, y_cal=y_cal,
        p_tst=p_tst, y_tst=y_tst,

        idx_tr=idx_tr, idx_sel=idx_sel, idx_cal=idx_cal,
        idx_train_pool=idx_train_pool,
        idx_test=idx_test,

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
    print("[counts] pool", int(counts_pool.sum()),
          "train", int(counts_tr.sum()),
          "sel", int(counts_sel.sum()),
          "cal", int(counts_cal.sum()),
          "test", int(counts_test.sum()))


if __name__ == "__main__":
    main()
