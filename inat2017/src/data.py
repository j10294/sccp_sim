# src/data.py
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

K_INAT2017 = 5089  # TFDS i_naturalist2017 classes :contentReference[oaicite:2]{index=2}

def load_tfds_inat(split: str, img_size=(299, 299), seed: int = 0, shuffle=True):
    """
    split: 'train' or 'validation' (TFDS naming)
    yields (image[0..1], label int32)
    """
    ds = tfds.load(
        "i_naturalist2017",
        split=split,
        data_dir=data_dir,          # None이면 TFDS default/TFDS_DATA_DIR 사용
        try_gcs=True,               # 핵심
        download_and_prepare_kwargs={"download_config": download_config},
    )

    def _pp(img, y):
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.cast(y, tf.int32)

    return ds.map(_pp, num_parallel_calls=tf.data.AUTOTUNE)

def take_n(ds, n: int):
    return ds.take(n)

def to_batched(ds, batch_size: int):
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def split_indices(n: int, frac_tr=0.4, frac_cw=0.2, frac_cq=0.2, frac_te=0.2, seed=0):
    assert abs((frac_tr+frac_cw+frac_cq+frac_te) - 1.0) < 1e-8
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = int(frac_tr * n)
    n_cw = int(frac_cw * n)
    n_cq = int(frac_cq * n)
    i_tr = idx[:n_tr]
    i_cw = idx[n_tr:n_tr+n_cw]
    i_cq = idx[n_tr+n_cw:n_tr+n_cw+n_cq]
    i_te = idx[n_tr+n_cw+n_cq:]
    return i_tr, i_cw, i_cq, i_te

def nested_split_indices(n: int, seed=0):
    """Split into two halves (cw1/cw2)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    mid = n // 2
    return idx[:mid], idx[mid:]

def save_npz(path: str, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **kwargs)

def load_npz(path: str):
    return np.load(path, allow_pickle=False)
