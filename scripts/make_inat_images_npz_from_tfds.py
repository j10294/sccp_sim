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

def iter_split(builder, split_name, data_dir, seed, shuffle_files=True):
    ds = tfds.load(
        builder.name,
        data_dir=data_dir,
        split=split_name,
        download=False,            # 재다운 방지
        shuffle_files=shuffle_files,
        as_supervised=False,
    )
    # TFRecord shard order shuffle은 shuffle_files로; sample order는 아래 shuffle로
    ds = ds.shuffle(10000, seed=seed, reshuffle_each_iteration=False)
    return ds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="TFDS data_dir (e.g., ~/sccp_sim/data/tfds)")
    ap.add_argument("--out_npz", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--max_train", type=int, default=0, help="0 means use full train split; else take first N after shuffle")
    ap.add_argument("--max_val", type=int, default=0, help="0 means use full validation split; else take first N after shuffle")
    ap.add_argument("--max_test", type=int, default=0, help="0 means use full test split; else take first N after shuffle")
    ap.add_argument("--no_shuffle", action="store_true", help="do not shuffle examples within each split")
    args = ap.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    out_npz = os.path.expanduser(args.out_npz)

    builder = tfds.builder("i_naturalist2017", data_dir=data_dir)
    info = builder.info
    splits = info.splits
    print("[tfds] splits:", {k: int(v.num_examples) for k, v in splits.items()})

    def load_one_split(split_name: str, max_n: int):
        n_total = int(splits[split_name].num_examples)
        n_take = n_total if max_n == 0 else min(max_n, n_total)
        print(f"[load] {split_name}: taking {n_take} / {n_total}")

        ds = tfds.load(
            "i_naturalist2017",
            data_dir=data_dir,
            split=split_name,
            download=False,
            shuffle_files=(not args.no_shuffle),
            as_supervised=False,
        )
        if not args.no_shuffle:
            ds = ds.shuffle(10000, seed=args.seed, reshuffle_each_iteration=False)

        ds = ds.take(n_take)

        X = np.empty((n_take, 3, args.image_size, args.image_size), dtype=np.uint8)
        y = np.empty((n_take,), dtype=np.int64)

        for i, ex in enumerate(tqdm(tfds.as_numpy(ds), total=n_take)):
            img = tf.convert_to_tensor(ex["image"])   # HWC uint8
            img = resize_uint8(img, args.image_size)
            x = img.numpy()                           # HWC
            X[i] = np.transpose(x, (2, 0, 1))         # CHW
            y[i] = int(ex["label"])

        return X, y

    X_train, y_train = load_one_split("train", args.max_train)
    X_val, y_val     = load_one_split("validation", args.max_val)
    X_test, y_test   = load_one_split("test", args.max_test)

    meta = {
        "tfds_data_dir": data_dir,
        "dataset": "i_naturalist2017",
        "version": str(info.version),
        "seed": int(args.seed),
        "image_size": int(args.image_size),
        "format": "uint8_CHW",
        "max_train": int(args.max_train),
        "max_val": int(args.max_val),
        "max_test": int(args.max_test),
        "note": "train/validation/test splits preserved from TFDS",
    }

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        meta=np.array([meta], dtype=object),
    )
    print(f"[saved] {out_npz}")
    print("[shapes]",
          "X_train", X_train.shape, "y_train", y_train.shape,
          "X_val",   X_val.shape,   "y_val",   y_val.shape,
          "X_test",  X_test.shape,  "y_test",  y_test.shape)

if __name__ == "__main__":
    main()
