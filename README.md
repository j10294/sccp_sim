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

## 1. Dependencies

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

### 실행 예시 (MobileNetV3-Small)
```bash
python scripts/train_and_export_probs_inat.py \
  --img_npz data/npz/inat2017_images_t50k_v10k_te10k_seed1.npz \
  --out_prob_npz data/npz/inat2017_probs_selA_calB_test_mnv3s_fz_t50k_ep5_seed1.npz \
  --model mobilenet_v3_small \
  --epochs 5 \
  --batch_size 128 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --calibA_frac 0.5 \
  --calib_split_seed 1 \
  --seed 1
```

### 출력 파일
```bash
data/npz/
└── inat2017_probs_selA_calB_test_mnv3s_fz_t50k_ep5_seed1.npz
```
이 확률 NPZ 파일은 이후의 Conformal Prediction 방법들의 input으로 사용된다. 

---

## 4. Conformal Prediction
### 공통 실행 예시
```bash
python run_cp_from_npz.py \
  --npz data/npz/inat2017_probs_selA_calB_test_mnv3s_fz_t50k_ep5_seed1.npz \
  --alpha 0.1 \
  --K 5089 \
  --method all
```
- `alpha` : miscoverage level (e.g. 0.05)
- `K` : number of class (iNat2017의 경우 5089)

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
python run_cp_from_npz.py \
  --npz data/npz/inat2017_probs_selA_calB_test_mnv3s_fz_t50k_ep5_seed1.npz \
  --alpha 0.1 \
  --K 5089 \
  --method sccp \
  --Kc 10 \
  --tail_frac 0.2
```
- `Kc` : number of clusters
- `tail_frac` : tail class의 비율

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

