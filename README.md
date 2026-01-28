# Shrinkage-Class-Clustered Conformal Prediction (SCC-CP)
본 저장소는 대규모 다중 클래스 이미지 분류 문제에서 **Global CP / Class-Conditional CP / Shrinkage Class-Clustered CP (SCCP)** 방법을 비교하기 위한 실험 코드 모음이다.

주요 실험 데이터셋은 **iNat2017**이며, 추가적으로 **CIFAR100** 데이터셋에 대해서도 실험하였다. 

---

## 전체 파이프라인 요약

1. iNat2017 이미지 데이터 → NPZ 변환  
2. 이미지 분류 모델 학습 및 **확률 출력 NPZ 생성**  
3. 확률 NPZ를 입력으로 conformal prediction 수행  
4. GCP / CCCP / SCCP 성능 비교 (coverage, set size 등)

---

## 0. Dependencies

The following Python packages are required to run the experiments:

- Python ≥ 3.9
- numpy
- scipy
- scikit-learn
- pandas
- matplotlib
- torch
- torchvision
- tqdm
- pillow
- tensorflow
- tensorflow-datasets

## 1. Dataset
Experiments are conducted on the iNaturalist 2017 (iNat2017) dataset,
a large-scale and highly imbalanced real-world image classification benchmark
[Van Horn et al., 2018].

iNat2017은 실제 자연 환경에서 수집된 대규모 이미지 분류 데이터셋으로, 클래스 간 샘플 수의 불균형이 매우 심한것이 특징이다. 일부 소수 클래스는 극히 적은 관측치를 가지지만, 다수의 클래스는 수천장 이상의 이미지를 포함하고 있어 long-tailed 분포를 잘 반영한다. 

이러한 특성으로 인해 GCP가 소수 클래스에 대해 과도하게 보수적이거나, 반대로 신뢰성을 보장하지 못하는 경우를 드러내기에 적합한 데이터라 생각된다. 본 실험에서는 iNat2017을 활용하여 클래스 불균형이 심한 상황에서도 우리가 제안하는 SCCP 방법이 기존 방법들에 비해 안정적이고 균형잡힌 prediction set을 제공할 수 있음을 검증하고자 한다. 


## 2. iNat2017 데이터를 NPZ로 변환
원본 파일은 용량 문제로 직접 사용하지 않으며, 실험에 필요한 train/val/test split을 포함한 NPZ 파일로 변환하여 사용한다.

### 실행 예시
```bash
python scripts/make_npz_inat2017_tfds_tfhub.py \
  --out_npz data/npz/inat2017_images_t50k_v10k_te10k_seed1.npz \
  --seed 1 \
  --n_sel 50000 \
  --n_cal 10000 \
  --n_test 10000
```

### 생성 파일
```bash
data/npz/
└── inat2017_images_t50k_v10k_te10k_seed1.npz
```

해당 NPZ에는 다음의 정보가 포함된다 : 
- 이미지 배열
- 정답 레이블
- selection / calibration / test index

---

## 3. 모델 학습 및 확률 NPZ 생성
앞서 변환한 이미지 NPZ 파일을 input으로 하고, 확률 NPZ 를 output으로 하는 모델을 학습한다. 이 단계의 목적은 각 이미지에 대한 class probability를 얻는 것이며, 모델의 분류 정확도 자체는 Conformal prediction 방법 특성상 크게 중요하지 않다. 그러나 정확도가 너무 낮은 모델의 경우에는 class probability가 거의 동일하여 자칫 prediction set size를 너무 크게 만들 수 있으므로, 정확도가 너무 낮은 모델을 사용하는 것에는 주의가 필요하다. 

### 실행 예시 (ResNet50, head-only)
```bash
python3 scripts/train_and_export_probs_inat.py \
  --img_npz data/npz/inat2017_images_strat_t50k_v30k_te10k_seed1.npz \
  --out_prob_npz data/npz/inat2017_probs_strat_selA_calB_test_rn50_head_t50k_ep10_seed1.npz \
  --model resnet50 \
  --finetune head \
  --epochs 10 \
  --batch_size 128 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --calib_split_seed 1 \
  --seed 1
```



### 옵션 설명
#### 입/출력 관련
- `--img_npz` : iNat2017 원본 이미지를 전처리하여 저장한 이미지 NPZ 파일 경로.
- `--out_prob_npz` : 학습된 모델로부터 얻은 class probability 결과를 저장할 NPZ 파일 경로. 생성되는 파일에는 다음 항목들이 포함된다. 
  -  `p_sel, y_sel` : selection set 확률 및 레이블
  -  `p_cal, y_cal` : calibration set 확률 및 레이블
  - `p_test, y_test` : test set 확률 및 레이블
  - `counts_pool` : train split 기준 클래스별 샘플 수 (tail 정의용)

#### 모델 및 전이학습 설정
- `--model`  : resnet18, resnet50, mobilenet_v3_small, efficientnet_b0, convnext_tiny
- `--finetune` : `head`는 classification head만 학습, backbone은 고정.`last` 는 마지막 stage+head만 학습,`full`은 전체 네트워크를 fine-tuning.

#### 학습 하이퍼파라미터
- `--epochs` : 학습 에폭 수
- `--batch_size` : 미니배치 크기. GPU에 맞게 조절 가능.
- `--lr` : 학습률
- `--weight_decay` : L2 정규화 계수. overfitting 방지 위해 사용.

##### Calibration Split
- `--calibA_frac` : validation set을 Calib-A / Calib-B / Test2로 나눌 때, Calib-A가 차지하는 비율.
  - Calib-A: selection set (clustering / embedding용)
  - Calib-B: calibration set (threshold 추정용)
- `--calib_split_seed` : validation set 분할에 사용되는 random seed.
동일 seed 사용 시 실험 재현 가능.

##### 재현성 및 연산 옵션
- `--seed` : 모델 초기화 및 데이터 로딩에 사용되는 random seed.
- `--amp` : Automatic Mixed Precision (FP16) 사용. GPU 메모리 사용량을 줄이고 학습 속도를 향상시킨다. (모델 출력 확률에는 영향을 주지 않음)

### 출력 파일
```bash
data/npz/
└── inat2017_probs_selA_calB_test_rn50_head_t50k_ep5_seed1.npz
```
이 확률 NPZ 파일은 이후의 Conformal Prediction 방법들의 input으로 사용된다. 

---

## 4. Conformal Prediction

### 비교 방법
- **GCP (Global Conformal Prediction)**  
  A baseline conformal prediction method using a global threshold  
  [Vovk et al., 2005; Angelopoulos et al., 2022].

- **CC-CP (Class-Conditional Conformal Prediction)**  
  Class-wise conformal thresholds  
  [Ding et al., 2023].

- **SCCP (Shrinkage-Clustered Conformal Prediction)**  
  Clustered and shrinkage-based conformal thresholds (ours).

---

## 5. SCCP 
### 5.1 Class Embedding
- calibration data에서 각 클래스별 score 수집
- score quantile embedding (Simulation) or mean embedding (iNat2017)
### 5.2 Clustering
- embedding vector에 대해 kmeans clustering 적용
- `Kc` : Cluster number (default = 10)
### 5.3 Shrinkage
- cluster 단위로 shrinkage parameter `lambda_hat` 학습
- tail class에 대한 prediction set size를 줄이는 것이 목적

### 실행 예시
```bash
python3 scripts/run_cp_from_npz.py \
  --npz data/npz/inat2017_probs_strat_selA_calB_test_rn50_head_t50k_ep20_seed1.npz \
  --alpha 0.1 \
  --K 5089 \
  --clusters 10 \
  --tau 30 \
  --embed score_quantile \
  --q_grid 0.5,0.6,0.7,0.8,0.9 \
  --seed 1
```
-  `npz` : 확률 예측값과 sel/cal/test split이 저장된 파일 경로
- `K` : 전체 클래스 수. (inat2017의 경우 `5089`)
-  `clusters ` : 클러스터 개수 
- `alpha` : 목표 miscoverage 수준
-  `tau` : shrinkage parameter
- `embed` : class embedding method
  - `score_quantile` : 클래스별 score 분포의 quantile 기반 임베딩 (default)
  - `prob_mean` : 평균 확률 벡터 기반 임베딩 
- `q_grid` : `score_quantile` 에서 사용하는 quantile grid


### (Note) Running iNat2017 training safely 
서버에 CUDA/cuDNN이 여러 버전 설치되어 있어 fine-tuning에서 오류가 발생하는 경우, 다음 스크립트를 통해 실행
```bash
chmod +x run_inat_full.sh
./run_inat_full.sh
```

---

## 6. 출력 및 평가 지표
기본 출력 지표는 다음과 같다. 
- Overall coverage
- Average prediction set size
- Classwise coverage
- Clusterwise coverage
- Shrinkage parameter distribution(`lambda_hat`)

예시 출력 : 
```bash
Overall coverage: 0.901
Avg set size: 512.4
Mean lambda: 0.38
```

---

## 8. TODO_0114

- CIFAR / iNat README 분리
- 실험 결과 테이블 자동 생성
- clusterwise 결과 시각화 스크립트 정리


# References
- Vovk, V., Gammerman, A., and Shafer, G. (2005).
*Algorithmic Learning in a Random World*.
Springer.

- Angelopoulos, A. N., Bates, S., Malik, J., and Jordan, M. I. (2022).
*Uncertainty Sets for Image Classifiers using Conformal Prediction*.
Advances in Neural Information Processing Systems (NeurIPS 2022).

- Van Horn, G., Mac Aodha, O., Song, Y., Cui, Y., Sun, C.,
  Shepard, A., Adam, H., Perona, P., and Belongie, S. (2018).
*The iNaturalist Species Classification and Detection Dataset*.
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2018).
