# Code overview

이 repository는 이미지 분류 모델의 예측확률을 NPZ로 저장한 뒤, 이를 입력으로 GCP/CCCP/SCCP 를 실행한 후 그 결과를 비교한다.
- `train_*.py` : 데이터 split + 모델 학습 + 확률 출력 (NPZ 생성)
- `run_cp_from_npz.py` : NPZ를 읽어 CP 실행 + 지표 출력 (+JSON 저장 옵션)


# Dataset settings
## Balanced CIFAR-100
- `train_cifar100.py` : balanced dataset 실험용
## Long-tailed CIFAR-100-LT 
- `train_cifar100_lt.py` : train/calibration 은 Long tail, test는 원본 CIFAR-100
  - long_tail 환경에서 calibration scarcity 문제가 생기는지 확인을 위한 설정
  - 출력 : `...cifar100LT_probs/*.npz`

# Conforaml Methods
- GCP : 전체 클래스를 pooling, global threshold 1개만 사용
- CCCP : (클러스터 기반) 클러스터별 threshold 사용
- SCCP : CCCP threshold를 global 쪽으로 shrinkage 해서 희소 클래스에서 안정화
  - shrinkage hyperparameter : `tau`
  - embedding : `--embed score_quantile` (클래스 score quantile 벡터 기반) -> kmeans clustering에 사용

# Running experiments
## 1. Model Training & Generating NPZ (CIFAR-100-LT)
```bash
python train_cifar100_lt.py \
  --data_root ./data \
  --out_dir ./out/cifar100LT_probs \
  --use_gpu \
  --lt --imb_type exp --imb_factor 100 \
  --split_mode fracs \
  --train_frac 0.6 --select_frac 0.2 --calib_frac 0.2 \
  --epochs 100 --batch_size 128 --lr 0.01 \
  --seed_data 1 --seed_train 1
  --arch resnet152 #["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d"]
```
- 생성되는 npz에는 보통 `p_sel, y_sel, p_cal, y_cal, p_tst, y_tst` 및 tail 관련 정보가 포함됨
- `seed_data`는 data split, `seed_train`은 train randomness에 대응됨.


## 2. GCP/CCCP/SCCP실행 
```bash
python3 run_cp_from_npz.py \
  --npz "/home/jisukim/sccp_sim/cifar100/out/cifar100LT_probs/cifar100_resnet101_e100_bs128_LT_exp_IF100_tailfrac0.2.npz" \
  --K 100 \
  --alpha 0.1 \
  --clusters 10 \
  --tau 50 \
  --seed 1 \
  --embed score_quantile
```
- 출력 : overall coverage / set size, tail_head coverage/ set size, classwise, clusterwise 요약 지표
- `--clusters`는 non_null cluster 개수이며, null cluster는 자동으로 추가될 수 있음 (샘플 희소 정도에 따라 결정)

## Hyperparameter Selection 
### tau sweep 
```bash
python run_cp_from_npz.py \
  --npz "/home/jisukim/sccp_sim/cifar100/out/cifar100LT_probs/cifar100_resnet50_e100_bs128_LT_exp_IF100_tailfrac0.2.npz" \
  --alpha 0.1 --K 100 --seed 1 \
  --sweep \
  --tau_list 0,1,5,10,20,50,100,200 \
  --clusters_list 5,10,20 \
  --embed score_quantile
```
- sweep 출력 결과를 보고 가장 안정적인 tau/cluster 조합 선택 후 단일 실행으로 결과를 저장. 


## Export to latex table ###
```bash
python scripts/json_to_latex_table.py \
  --json %앞에서 설정한 json 파일명 \
  --single_block \
  --caption "CIFAR-100-LT (IF=100), ResNet-18 (100 epochs)." \
  --label "tab:cifar100lt_resnet18_e100_tau5" \ #모델명, tau 수정 
  --out_tex tables/cifar100lt_resnet18_e100_tau5.tex #모델명, tau 수정 
```
- 논문에 바로 붙일 수 있는 형태로 테이블 생성
- `caption/label/out_tex`는 모델명/epoch/tau에 맞게 수정.

## Notes/TODO
- `train_cifar100_scc_raps.py` : balanced dataset에서의 SCCP_RAPS 실험용. RAPS 방법을 적용해 SCCP set size를 줄일 수 있는지 확인 필요. 
- long-tail experiments에서는 class별 calibration sample size가 매우 작을 수 있어 CCCP가 불안정해짐.SCCP는 이를 tau shrinkage로 완화하는 것이 목적.

