# scripts/make_probs.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import argparse, numpy as np, tensorflow as tf
from src.data import load_npz, save_npz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_npz", type=str, required=True)     # F,y
    ap.add_argument("--head_ckpt", type=str, required=True)    # tf savedmodel
    ap.add_argument("--out_npz", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)  # >1 flattens -> worse
    ap.add_argument("--logit_noise", type=float, default=0.0)  # add N(0, sigma^2) to logits
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = load_npz(args.feat_npz)
    F, y = data["F"], data["y"]

    head = tf.keras.models.load_model(args.head_ckpt)

    rng = np.random.default_rng(args.seed)
    Ps = []
    for i in range(0, len(y), args.batch_size):
        fb = F[i:i+args.batch_size]
        logits = head(fb).numpy()
        if args.logit_noise > 0:
            logits = logits + rng.normal(0.0, args.logit_noise, size=logits.shape)
        logits = logits / args.temperature
        p = tf.nn.softmax(logits, axis=1).numpy()
        Ps.append(p)
    P = np.concatenate(Ps, axis=0)

    save_npz(args.out_npz, P=P, y=y)
    print("saved", args.out_npz, P.shape, y.shape)

if __name__ == "__main__":
    main()
