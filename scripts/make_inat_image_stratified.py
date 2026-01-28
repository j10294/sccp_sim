import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

def resize_uint8(img: tf.Tensor, size: int) -> tf.Tensor:
    """
    img: HWC uint8 (or convertible)
    returns: HWC uint8 resized to (size,size)
    """
    if img.dtype != tf.uint8:
        img = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8)
    img = tf.image.resize(img, (size, size), method="bilinear")
    img = tf.clip_by_value(img, 0, 255)
    return tf.cast(img, tf.uint8)

def load_pool_dataset(data_dir: str, split_expr: str):
    # IMPORTANT: keep deterministic order across passes
    ds = tfds.load(
        "i_naturalist2017",
        data_dir=data_dir,
        split=split_expr,          # e.g., "train+validation"
        download=False,
        shuffle_files=False,       # must be False for stable indexing
        as_supervised=False,
    )
    return ds

def scan_labels(ds):
    """
    First pass: scan labels only and return labels array.
    Deterministic order required across passes.
    """
    labels = []
    for ex in tqdm(tfds.as_numpy(ds), desc="[pass1] scanning labels"):
        labels.append(int(ex["label"]))
    return np.asarray(labels, dtype=np.int64)

def allocate_counts_proportional(class_sizes, total_target, min_per_class=0, rng=None):
    """
    Allocate integer counts per class summing to total_target approximately proportional to class_sizes.

    Enforces:
    - counts[c] <= class_sizes[c]
    - counts[c] >= min_per_class if possible (i.e., if class_sizes[c] >= min_per_class)
    - sum(counts) == total_target (or best effort if impossible)

    Returns: counts array (len = num_classes)
    """
    K = len(class_sizes)
    sizes = np.asarray(class_sizes, dtype=np.int64)

    # Feasibility check
    total_available = int(sizes.sum())
    if total_target > total_available:
        raise ValueError(f"total_target={total_target} > total_available={total_available}")

    counts = np.zeros(K, dtype=np.int64)

    # Apply minimums where feasible
    if min_per_class > 0:
        feasible = sizes >= min_per_class
        counts[feasible] = min_per_class
        remaining = total_target - int(counts.sum())
        if remaining < 0:
            # Can't even satisfy mins
            # Back off: drop mins and do pure proportional
            counts[:] = 0
            remaining = total_target
    else:
        remaining = total_target

    capacity = sizes - counts
    if remaining == 0:
        return counts

    # Proportional allocation on remaining capacity
    weights = sizes.astype(np.float64)
    wsum = weights.sum()
    if wsum <= 0:
        # Degenerate
        # Fill arbitrarily
        idx = np.where(capacity > 0)[0]
        if rng is None:
            rng = np.random.default_rng(0)
        rng.shuffle(idx)
        for c in idx:
            take = min(int(capacity[c]), remaining)
            counts[c] += take
            remaining -= take
            if remaining == 0:
                break
        return counts

    raw = remaining * (weights / wsum)
    add_floor = np.floor(raw).astype(np.int64)

    # Respect capacity
    add = np.minimum(add_floor, capacity)
    counts += add
    remaining -= int(add.sum())
    capacity = sizes - counts

    if remaining == 0:
        return counts

    # Distribute remainder by largest fractional parts, respecting capacity
    frac = raw - np.floor(raw)
    # Break ties randomly but deterministically if rng is provided
    order = np.argsort(-frac)  # descending frac
    if rng is not None:
        # small random jitter for tie-breaking
        jitter = rng.random(K) * 1e-12
        order = np.argsort(-(frac + jitter))

    for c in order:
        if remaining == 0:
            break
        if capacity[c] > 0:
            counts[c] += 1
            capacity[c] -= 1
            remaining -= 1

    # If still remaining (because many classes hit capacity), fill anywhere with capacity
    if remaining > 0:
        idx = np.where(capacity > 0)[0]
        if rng is None:
            rng = np.random.default_rng(0)
        rng.shuffle(idx)
        for c in idx:
            if remaining == 0:
                break
            take = min(int(capacity[c]), remaining)
            counts[c] += take
            remaining -= take

    return counts

def stratified_split_indices(labels, n_train, n_calib, n_test, seed=1,
                            min_test_per_class=1, min_calib_per_class=1):
    """
    Create stratified indices for train/calib/test with fixed totals.

    Strategy:
    - For each class c, shuffle its indices.
    - Decide how many to put in test and calib (with minimums when feasible).
    - Remaining goes to train.
    - Ensures exact totals if feasible.

    Note:
    - Classes with very small counts may not satisfy both min_test_per_class and min_calib_per_class.
      We enforce mins only when class_size >= (min_test + min_calib + 1), so at least 1 left for train.
    """
    rng = np.random.default_rng(seed)

    labels = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    K = len(classes)

    # map from class value to 0..K-1
    class_to_k = {int(c): i for i, c in enumerate(classes)}
    sizes = counts.copy()

    total = int(labels.shape[0])
    target_total = n_train + n_calib + n_test
    if target_total > total:
        raise ValueError(f"Requested {target_total} samples but only {total} available in pool.")

    # Determine per-class min feasibility for (test, calib) while leaving >=1 for train
    min_tc = min_test_per_class + min_calib_per_class + 1
    feasible_tc = sizes >= min_tc

    # Allocate test counts per class
    test_counts = allocate_counts_proportional(
        class_sizes=sizes,
        total_target=n_test,
        min_per_class=0,   # we'll apply mins manually via feasibility logic
        rng=rng
    )
    # Impose min_test where feasible
    if min_test_per_class > 0:
        for k in np.where(feasible_tc)[0]:ㅇ
            if test_counts[k] < min_test_per_class:
                # steal from a class with surplus test allocation
                need = min_test_per_class - int(test_counts[k])
                if need <= 0:
                    continue
                donors = np.where(test_counts > min_test_per_class)[0]
                for d in donors:
                    take = min(int(test_counts[d] - min_test_per_class), need)
                    if take > 0:
                        test_counts[d] -= take
                        test_counts[k] += take
                        need -= take
                    if need == 0:
                        break

    # Now allocate calib counts from remaining capacity
    remaining_after_test = sizes - test_counts
    # To leave >=1 for train for feasible classes, cap calib to remaining-1 there
    cap_for_calib = remaining_after_test.copy()
    cap_for_calib[feasible_tc] = np.maximum(0, cap_for_calib[feasible_tc] - 1)

    calib_counts = np.zeros(K, dtype=np.int64)
    # First, apply min_calib where feasible and possible
    if min_calib_per_class > 0:
        for k in np.where(feasible_tc)[0]:
            if cap_for_calib[k] >= min_calib_per_class:
                calib_counts[k] = min_calib_per_class

    remaining_calib = n_calib - int(calib_counts.sum())
    if remaining_calib < 0:
        # Can't satisfy min calib; back off
        calib_counts[:] = 0
        remaining_calib = n_calib

    # Proportionally allocate the rest of calib within cap_for_calib
    if remaining_calib > 0:
        # weights proportional to remaining_after_test (or sizes)
        weights = remaining_after_test.astype(np.float64)
        wsum = weights.sum()
        if wsum <= 0:
            # fill arbitrarily
            idx = np.where(cap_for_calib > calib_counts)[0]
            rng.shuffle(idx)
            for k in idx:
                if remaining_calib == 0:
                    break
                can = int(cap_for_calib[k] - calib_counts[k])
                take = min(can, remaining_calib)
                calib_counts[k] += take
                remaining_calib -= take
        else:
            cap_left = cap_for_calib - calib_counts
            raw = remaining_calib * (weights / wsum)
            add_floor = np.floor(raw).astype(np.int64)
            add = np.minimum(add_floor, cap_left)
            calib_counts += add
            remaining_calib -= int(add.sum())
            cap_left = cap_for_calib - calib_counts

            if remaining_calib > 0:
                frac = raw - np.floor(raw)
                jitter = rng.random(K) * 1e-12
                order = np.argsort(-(frac + jitter))
                for k in order:
                    if remaining_calib == 0:
                        break
                    if cap_left[k] > 0:
                        calib_counts[k] += 1
                        cap_left[k] -= 1
                        remaining_calib -= 1

                if remaining_calib > 0:
                    idx = np.where(cap_left > 0)[0]
                    rng.shuffle(idx)
                    for k in idx:
                        if remaining_calib == 0:
                            break
                        take = min(int(cap_left[k]), remaining_calib)
                        calib_counts[k] += take
                        remaining_calib -= take

    # Train counts are whatever is left, but we only need n_train total:
    train_counts = sizes - test_counts - calib_counts
    if int(train_counts.sum()) < n_train:
        raise ValueError("Not enough remaining samples for train after allocating test+calib.")
    # If train_counts.sum() > n_train, we will subsample within train per class proportionally
    if int(train_counts.sum()) > n_train:
        train_counts = allocate_counts_proportional(
            class_sizes=train_counts,
            total_target=n_train,
            min_per_class=0,
            rng=rng
        )

    # Now actually pick indices per class
    idx_train, idx_calib, idx_test = [], [], []

    for k, c in enumerate(classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)

        n_t = int(test_counts[k])
        n_c = int(calib_counts[k])
        n_r = int(train_counts[k])

        # order: test | calib | train  (arbitrary, but deterministic)
        # ensure we don't exceed available
        n_total_needed = n_t + n_c + n_r
        if n_total_needed > len(idx_c):
            # fallback: trim train first
            overflow = n_total_needed - len(idx_c)
            n_r = max(0, n_r - overflow)

        idx_test.extend(idx_c[:n_t])
        idx_calib.extend(idx_c[n_t:n_t + n_c])
        idx_train.extend(idx_c[n_t + n_c:n_t + n_c + n_r])

    # Final shuffle within each split for good measure (deterministic)
    idx_train = np.asarray(idx_train, dtype=np.int64)
    idx_calib = np.asarray(idx_calib, dtype=np.int64)
    idx_test  = np.asarray(idx_test,  dtype=np.int64)

    rng.shuffle(idx_train)
    rng.shuffle(idx_calib)
    rng.shuffle(idx_test)

    # Sanity checks
    assert len(np.intersect1d(idx_train, idx_calib)) == 0
    assert len(np.intersect1d(idx_train, idx_test)) == 0
    assert len(np.intersect1d(idx_calib, idx_test)) == 0

    return idx_train, idx_calib, idx_test, classes, sizes, train_counts, calib_counts, test_counts

def collect_by_indices(ds, take_indices_set, image_size, n_take, desc):
    """
    Second pass: iterate ds in deterministic order and keep only indices in take_indices_set.
    """
    X = np.empty((n_take, 3, image_size, image_size), dtype=np.uint8)
    y = np.empty((n_take,), dtype=np.int64)

    j = 0
    for i, ex in enumerate(tqdm(tfds.as_numpy(ds), desc=desc)):
        if i not in take_indices_set:
            continue
        img = tf.convert_to_tensor(ex["image"])  # HWC uint8
        img = resize_uint8(img, image_size)
        x = img.numpy()
        X[j] = np.transpose(x, (2, 0, 1))
        y[j] = int(ex["label"])
        j += 1
        if j == n_take:
            break

    if j != n_take:
        raise RuntimeError(f"Collected {j}/{n_take} samples for {desc}. Order changed or indices mismatch.")
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="TFDS data_dir (e.g., ~/sccp_sim/data/tfds)")
    ap.add_argument("--out_npz", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--image_size", type=int, default=224)

    ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--n_calib", type=int, default=30000)
    ap.add_argument("--n_test",  type=int, default=10000)

    ap.add_argument("--pool_split", type=str, default="train+validation",
                    help='TFDS split expression for labeled pool (default: "train+validation")')

    ap.add_argument("--min_test_per_class", type=int, default=1)
    ap.add_argument("--min_calib_per_class", type=int, default=1)

    ap.add_argument("--save_indices", action="store_true",
                    help="also store split indices (pool indices) in the NPZ")

    args = ap.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    out_npz = os.path.expanduser(args.out_npz)

    builder = tfds.builder("i_naturalist2017", data_dir=data_dir)
    info = builder.info
    print("[tfds] version:", info.version)
    print("[tfds] splits:", {k: int(v.num_examples) for k, v in info.splits.items()})
    print("[pool] using split expression:", args.pool_split)

    ds_pool_1 = load_pool_dataset(data_dir, args.pool_split)
    labels = scan_labels(ds_pool_1)
    print("[pass1] pool size:", labels.shape[0], "num_classes:", len(np.unique(labels)))

    idx_tr, idx_cal, idx_te, classes, sizes, tr_ct, cal_ct, te_ct = stratified_split_indices(
        labels=labels,
        n_train=args.n_train,
        n_calib=args.n_calib,
        n_test=args.n_test,
        seed=args.seed,
        min_test_per_class=args.min_test_per_class,
        min_calib_per_class=args.min_calib_per_class,
    )

    print("[split sizes]", "train", len(idx_tr), "calib", len(idx_cal), "test", len(idx_te))

    # Second pass: must recreate dataset to restart iterator, same deterministic order.
    ds_pool_2 = load_pool_dataset(data_dir, args.pool_split)

    set_tr = set(map(int, idx_tr.tolist()))
    set_cal = set(map(int, idx_cal.tolist()))
    set_te = set(map(int, idx_te.tolist()))

    X_train, y_train = collect_by_indices(ds_pool_2, set_tr, args.image_size, args.n_train, "[pass2] collect train")

    # Need a fresh iterator again for calib
    ds_pool_3 = load_pool_dataset(data_dir, args.pool_split)
    X_calib, y_calib = collect_by_indices(ds_pool_3, set_cal, args.image_size, args.n_calib, "[pass2] collect calib")

    # Fresh iterator again for test
    ds_pool_4 = load_pool_dataset(data_dir, args.pool_split)
    X_test, y_test = collect_by_indices(ds_pool_4, set_te, args.image_size, args.n_test, "[pass2] collect test")

    meta = {
        "tfds_data_dir": data_dir,
        "dataset": "i_naturalist2017",
        "tfds_version": str(info.version),
        "pool_split": args.pool_split,
        "seed": int(args.seed),
        "image_size": int(args.image_size),
        "format": "uint8_CHW",
        "n_train": int(args.n_train),
        "n_calib": int(args.n_calib),
        "n_test": int(args.n_test),
        "min_test_per_class": int(args.min_test_per_class),
        "min_calib_per_class": int(args.min_calib_per_class),
        "note": "Stratified sampling from labeled TFDS pool (train+validation by default). Deterministic TFDS order (shuffle_files=False); shuffling done in NumPy index space.",
    }

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)

    save_kwargs = dict(
        X_train=X_train, y_train=y_train,
        X_calib=X_calib, y_calib=y_calib,
        X_test=X_test,   y_test=y_test,
        meta=np.array([meta], dtype=object),
    )

    if args.save_indices:
        save_kwargs.update(
            idx_train=idx_tr,
            idx_calib=idx_cal,
            idx_test=idx_te,
            class_values=classes.astype(np.int64),
            class_sizes=sizes.astype(np.int64),
            class_counts_train=tr_ct.astype(np.int64),
            class_counts_calib=cal_ct.astype(np.int64),
            class_counts_test=te_ct.astype(np.int64),
        )

    np.savez_compressed(out_npz, **save_kwargs)

    print(f"[saved] {out_npz}")
    print("[shapes]",
          "X_train", X_train.shape, "y_train", y_train.shape,
          "X_calib", X_calib.shape, "y_calib", y_calib.shape,
          "X_test",  X_test.shape,  "y_test",  y_test.shape)

if __name__ == "__main__":
    main()
