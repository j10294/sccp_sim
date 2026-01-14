# scripts/extract_features_tfhub.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


import argparse, numpy as np, tensorflow as tf, tensorflow_hub as hub
from src.data import load_npz, save_npz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", type=str, required=True)   # contains X,y
    ap.add_argument("--out_npz", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hub_url", type=str, required=True)
    args = ap.parse_args()

    data = load_npz(args.in_npz)
    X, y = data["X"], data["y"]

    # TFHub feature extractor
    m = hub.KerasLayer(args.hub_url, trainable=False)

    feats = []
    for i in range(0, len(X), args.batch_size):
        xb = X[i:i+args.batch_size]
        fb = m(xb).numpy()
        feats.append(fb)
    F = np.concatenate(feats, axis=0)

    save_npz(args.out_npz, F=F, y=y)
    print("saved", args.out_npz, F.shape, y.shape)

if __name__ == "__main__":
    main()
