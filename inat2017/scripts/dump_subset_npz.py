# scripts/dump_subset_npz.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import argparse, numpy as np
from src.data import load_tfds_inat, take_n, to_batched, save_npz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train")     # train/validation
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    ds = load_tfds_inat(args.split, seed=args.seed, shuffle=True)
    ds = take_n(ds, args.n)
    ds = to_batched(ds, args.batch_size)

    X_list, y_list = [], []
    for xb, yb in ds:
        X_list.append(xb.numpy())
        y_list.append(yb.numpy())
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    save_npz(args.out, X=X, y=y)
    print("saved", args.out, X.shape, y.shape)

if __name__ == "__main__":
    main()
