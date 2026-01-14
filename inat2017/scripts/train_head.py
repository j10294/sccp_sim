# scripts/train_head.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, numpy as np, tensorflow as tf
from src.data import load_npz, save_npz, K_INAT2017

def build_head(d, K=K_INAT2017):
    inp = tf.keras.Input(shape=(d,))
    out = tf.keras.layers.Dense(K)(inp)  # logits
    return tf.keras.Model(inp, out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_npz", type=str, required=True)  # F,y
    ap.add_argument("--out_ckpt", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--train_limit", type=int, default=0)  # 0 means use all
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    data = load_npz(args.feat_npz)
    F, y = data["F"], data["y"]

    if args.train_limit and args.train_limit < len(y):
        idx = np.random.permutation(len(y))[:args.train_limit]
        F, y = F[idx], y[idx]

    d = F.shape[1]
    model = build_head(d)

    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=opt, loss=loss, metrics=["sparse_categorical_accuracy"])
    model.fit(F, y, batch_size=args.batch_size, epochs=args.epochs, verbose=1)

    model.save(args.out_ckpt)
    print("saved head:", args.out_ckpt)

if __name__ == "__main__":
    main()
