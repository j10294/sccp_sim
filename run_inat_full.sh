#!/usr/bin/env bash
set -euo pipefail

# --- project root로 이동 (상대경로 안전) ---
cd /data1/jisukim/sccp_sim

# --- venv python 절대경로 고정 ---
PY="/home/jisukim/sccp_sim/venv/bin/python3"

# --- 실행 전 환경 sanity check ---
echo "[env] python = $PY"
LD_LIBRARY_PATH= "$PY" - <<'PY'
import sys, os, torch
print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version())
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
PY

# --- 반복 실험 ---
for SD in 1 2 3; do
  for EP in 5 10 20; do
    IMG="data/npz/inat2017_images_strat_t50k_v30k_te10k_seed${SD}.npz"
    OUT="data/npz/inat2017_probs_strat_selA_calB_test_rn50_full_t50k_ep${EP}_seed${SD}.npz"

    echo "[run] SD=${SD} EP=${EP}"
    echo "      IMG=${IMG}"
    echo "      OUT=${OUT}"

    # 핵심: 이 커맨드 실행 동안만 LD_LIBRARY_PATH를 비움 (cuda-11.8 차단)
    LD_LIBRARY_PATH= "$PY" scripts/train_and_export_probs_inat.py \
      --img_npz "${IMG}" \
      --out_prob_npz "${OUT}" \
      --model resnet50 \
      --finetune full \
      --epochs "${EP}" \
      --batch_size 128 \
      --lr 1e-3 \
      --weight_decay 1e-4 \
      --calib_split_seed "${SD}" \
      --seed "${SD}"
  done
done
