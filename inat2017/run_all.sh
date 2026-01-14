#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =========================
# User config (edit here)
# =========================
SEED=1

# Subset sizes (start small, then scale up)
N_TRAIN=200
N_VAL=100

# Splits for simulation inside val-probs (these are in src/run_sim driver, not here)
# alpha for CP
ALPHA=0.1
N_CLUSTERS=50
N_SEEDS=3   # simulation seeds loop (you can raise later)

# Batching
BATCH_IMG=64
BATCH_FEAT=64
BATCH_HEAD=256
BATCH_PROB=512

# Head training knobs (Strong/Mid/Worst)
STRONG_EPOCHS=8
STRONG_TRAIN_LIMIT=0
STRONG_LR=1e-2

MID_EPOCHS=2
MID_TRAIN_LIMIT=20000
MID_LR=1e-2

WORST_EPOCHS=1
WORST_TRAIN_LIMIT=3000
WORST_LR=5e-3

# Optional degradation when generating probs
# (applied on top of the head; use to widen performance spread)
WORST_TEMP=2.0
WORST_NOISE=0.5

MID_TEMP=1.2
MID_NOISE=0.1

STRONG_TEMP=1.0
STRONG_NOISE=0.0

# TFHub / Kaggle model URL for iNat InceptionV3 feature vector
# (You may need to adjust if your TFHub loader requires a tfhub.dev URL.)
HUB_URL="https://www.kaggle.com/models/google/inception-v3/tensorFlow1/inaturalist-inception-v3-feature-vector"

# Paths
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_CACHE="${ROOT}/out/cache"
OUT_MODELS="${ROOT}/out/models"
OUT_RESULTS="${ROOT}/out/results"

mkdir -p "${OUT_CACHE}" "${OUT_MODELS}" "${OUT_RESULTS}"

TRAIN_RAW="${OUT_CACHE}/inat_train${N_TRAIN}_s${SEED}.npz"
VAL_RAW="${OUT_CACHE}/inat_val${N_VAL}_s${SEED}.npz"
TRAIN_FEAT="${OUT_CACHE}/inat_train${N_TRAIN}_s${SEED}_feat.npz"
VAL_FEAT="${OUT_CACHE}/inat_val${N_VAL}_s${SEED}_feat.npz"

HEAD_STRONG="${OUT_MODELS}/head_strong_s${SEED}"
HEAD_MID="${OUT_MODELS}/head_mid_s${SEED}"
HEAD_WORST="${OUT_MODELS}/head_worst_s${SEED}"

VAL_PROB_STRONG="${OUT_CACHE}/val${N_VAL}_strong_probs_s${SEED}.npz"
VAL_PROB_MID="${OUT_CACHE}/val${N_VAL}_mid_probs_s${SEED}.npz"
VAL_PROB_WORST="${OUT_CACHE}/val${N_VAL}_worst_probs_s${SEED}.npz"

RESULT_CSV="${OUT_RESULTS}/sim_val${N_VAL}_s${SEED}_alpha${ALPHA}_C${N_CLUSTERS}.csv"

echo "[1/5] Dump TFDS subsets -> NPZ"
python "${ROOT}/scripts/dump_subset_npz.py" \
  --split train --n "${N_TRAIN}" --seed "${SEED}" \
  --out "${TRAIN_RAW}" --batch_size "${BATCH_IMG}"

python "${ROOT}/scripts/dump_subset_npz.py" \
  --split validation --n "${N_VAL}" --seed "${SEED}" \
  --out "${VAL_RAW}" --batch_size "${BATCH_IMG}"

echo "[2/5] Extract features via TFHub -> NPZ(F,y)"
python "${ROOT}/scripts/extract_features_tfhub.py" \
  --in_npz "${TRAIN_RAW}" --out_npz "${TRAIN_FEAT}" \
  --batch_size "${BATCH_FEAT}" --hub_url "${HUB_URL}"

python "${ROOT}/scripts/extract_features_tfhub.py" \
  --in_npz "${VAL_RAW}" --out_npz "${VAL_FEAT}" \
  --batch_size "${BATCH_FEAT}" --hub_url "${HUB_URL}"

echo "[3/5] Train heads (Strong/Mid/Worst) on train features"
python "${ROOT}/scripts/train_head.py" \
  --feat_npz "${TRAIN_FEAT}" --out_ckpt "${HEAD_STRONG}" \
  --epochs "${STRONG_EPOCHS}" --batch_size "${BATCH_HEAD}" \
  --lr "${STRONG_LR}" --train_limit "${STRONG_TRAIN_LIMIT}" --seed "${SEED}"

python "${ROOT}/scripts/train_head.py" \
  --feat_npz "${TRAIN_FEAT}" --out_ckpt "${HEAD_MID}" \
  --epochs "${MID_EPOCHS}" --batch_size "${BATCH_HEAD}" \
  --lr "${MID_LR}" --train_limit "${MID_TRAIN_LIMIT}" --seed "${SEED}"

python "${ROOT}/scripts/train_head.py" \
  --feat_npz "${TRAIN_FEAT}" --out_ckpt "${HEAD_WORST}" \
  --epochs "${WORST_EPOCHS}" --batch_size "${BATCH_HEAD}" \
  --lr "${WORST_LR}" --train_limit "${WORST_TRAIN_LIMIT}" --seed "${SEED}"

echo "[4/5] Make probabilities on validation features (with optional degradation)"
python "${ROOT}/scripts/make_probs.py" \
  --feat_npz "${VAL_FEAT}" --head_ckpt "${HEAD_STRONG}" \
  --out_npz "${VAL_PROB_STRONG}" --batch_size "${BATCH_PROB}" \
  --temperature "${STRONG_TEMP}" --logit_noise "${STRONG_NOISE}" --seed "${SEED}"

python "${ROOT}/scripts/make_probs.py" \
  --feat_npz "${VAL_FEAT}" --head_ckpt "${HEAD_MID}" \
  --out_npz "${VAL_PROB_MID}" --batch_size "${BATCH_PROB}" \
  --temperature "${MID_TEMP}" --logit_noise "${MID_NOISE}" --seed "${SEED}"

python "${ROOT}/scripts/make_probs.py" \
  --feat_npz "${VAL_FEAT}" --head_ckpt "${HEAD_WORST}" \
  --out_npz "${VAL_PROB_WORST}" --batch_size "${BATCH_PROB}" \
  --temperature "${WORST_TEMP}" --logit_noise "${WORST_NOISE}" --seed "${SEED}"

echo "[5/5] Run simulation (CCCP vs SCC-CP vs Global) on probs"


python "${ROOT}/scripts/run_sim_cli.py" \
  --prob_npz_strong "${VAL_PROB_STRONG}" \
  --prob_npz_mid "${VAL_PROB_MID}" \
  --prob_npz_worst "${VAL_PROB_WORST}" \
  --alpha "${ALPHA}" --n_clusters "${N_CLUSTERS}" \
  --n_seeds "${N_SEEDS}" --out_csv "${RESULT_CSV}"

echo "DONE. Results: ${RESULT_CSV}"
