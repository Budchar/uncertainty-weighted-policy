# Safe Policy Deployment for Continuous Treatment Optimization

연속 처치(continuous treatment) 최적화에서 안전한 정책 배포 프레임워크

---

## 연구 개요

관찰 데이터에서 추정한 최적 가격 정책은 내생성(endogeneity)이나 데이터 부족으로 인해 실제와 다를 수 있습니다. 이 프로젝트는 추정의 불확실성을 개인 단위로 정량화하고, 불확실할수록 기존 정책(baseline)에 가깝게 보간하여 안전하게 배포하는 프레임워크를 구현합니다.

**핵심 공식**

```
π_safe(x) = (1 - λ·u(x)) · π*(x) + λ·u(x) · π₀(x)
```

- `π*(x)` — AI 추정 최적 정책 (Stage 1에서 산출)
- `π₀(x)` — Baseline 정책 (현재 운영 중인 고정 마진, 기본값 15%)
- `u(x)` — 개인별 종합 불확실성 (Stage 2에서 산출, 0~1)
- `λ` — 전역 보수성 파라미터 (0 = 보수성 없음, ≥1 = 강한 보수성)


## 프로젝트 구조

```
safe_policy/
├── configs/
│   └── default_config.py           # DGP 파라미터, 실험 시나리오 설정
├── src/
│   ├── stage0_dgp.py               # Stage 0: Semi-synthetic 데이터 생성
│   ├── stage1_estimate.py          # Stage 1: Dose-response curve 추정
│   ├── stage2_quantify.py          # Stage 2: 불확실성 정량화
│   └── stage3_deploy.py            # Stage 3: 안전한 정책 배포
├── utils/
│   ├── cache_utils.py              # Stage별 캐시 관리
│   └── gpu_utils.py                # XGBoost/GPU 가속 팩토리
├── run.py                          # 단일 파이프라인 실행 (캐시 지원)
├── run_experiment.py               # 체계적 비교 실험 (3축 × 반복)
├── run_detailed_analysis.py        # 5.3절 심층 분석 (상품별 상세)
├── analyze_synthesis_experiment.py  # Synthesis 방법 비교 분석
├── data_loader.py                  # M5 데이터 다운로드 유틸리티
├── m5_data/                        # M5 원시 데이터 (선택)
├── cache/                          # Stage별 캐시 파일
└── results/                        # 실험 결과 (JSON, CSV)
```


## 파이프라인 구성

### Stage 0: Semi-Synthetic DGP

M5 Walmart 판매 데이터의 현실적 covariate 구조 위에 경제학 기반 인과 메커니즘을 결합한 반합성(semi-synthetic) 데이터를 생성합니다. M5 데이터가 없으면 유사 구조의 synthetic covariates로 대체됩니다.

- 수요 모델: 지수감소 수요 `Q(m) = a · exp(-α · m)`, 최적 마진 `m* = 1/α`
- 내생성: 미관측 수요 신호가 가격과 판매 양쪽에 비슷한 스케일로 영향을 줍니다
- Ground truth(Oracle) 계산이 포함되어 있어, 모든 정책을 공정하게 비교할 수 있습니다

### Stage 1: Dose-Response Estimation

관찰 데이터에서 마진율→수익의 인과적 dose-response 관계를 추정합니다.

- GPS Regression (Hirano & Imbens 2004) 또는 DML (Chernozhukov et al. 2018)
- Bootstrap으로 개인별 신뢰구간을 산출하여 Stage 2의 입력으로 사용합니다

### Stage 2: Uncertainty Quantification

추정의 불확실성을 개인 단위로 정량화합니다. 두 축의 epistemic uncertainty를 측정하고 상호작용으로 합성합니다.

- `u_stat` — 통계적 불확실성: bootstrap 최적 treatment의 표준편차 (추정이 흔들리는 정도)
- `u_pos` — positivity 불확실성: GPS 기반 외삽 위험 (관찰이 부족한 정도)
- percentile 정규화로 스케일을 통일한 후 interaction 합성: `u(x) = (stat_p + pos_p + stat_p·pos_p) / 3`
- Adaptive Conformal Prediction으로 분포 가정 없는 개인별 예측 구간을 생성합니다

### Stage 3: Safe Policy Deployment

5개 정책을 비교 평가합니다.

1. **Naive Optimal** — 추정 최적을 그대로 적용합니다 (보수성 없음)
2. **Uniform Conservative** — 모든 개인에 동일한 보수성을 적용합니다
3. **Threshold-based** — 불확실성 임계값 초과 시 baseline으로 전환합니다
4. **Uncertainty-Weighted (제안 방법)** — 불확실성에 비례한 연속적 보간을 수행합니다
5. **Oracle** — ground truth 최적입니다 (성능 상한)


## 실행 방법

### 기본 파이프라인 (단일 시나리오)

```bash
# 기본 설정으로 전체 파이프라인 실행
python run.py

# 파라미터 지정
python run.py --n-samples 5000 --endogeneity 0.5 --estimator dml

# 특정 Stage부터 재실행
python run.py --from-stage 2

# 캐시 무시하고 처음부터
python run.py --force
```

### 체계적 비교 실험

```bash
# 빠른 테스트 (5~10분)
python run_experiment.py --quick

# 단일 축만 실험
python run_experiment.py --axis endogeneity --repeats 5

# 전체 실험 (4×3×3 시나리오 × 10반복 = 360회)
python run_experiment.py

# 중단 후 이어서 실행
python run_experiment.py --resume
```

### 심층 분석

```bash
# 상품별 상세 데이터 추출 (대표 시나리오 3개)
python run_detailed_analysis.py
```

### M5 데이터 준비 (선택)

M5 데이터 없이도 synthetic covariates로 동작합니다. 실제 M5 데이터를 사용하시려면 아래와 같이 진행하시면 됩니다.

```bash
# Kaggle API 설정 후
python data_loader.py

# 또는 수동으로 m5_data/ 폴더에 배치하시면 됩니다:
#   sales_train_evaluation.csv, sell_prices.csv, calendar.csv
```


## 의존성

### 필수

```
numpy
pandas
scikit-learn
scipy
```

### 권장 (성능 향상)

```
xgboost          # GPU 가속을 지원하며, sklearn GBR 대비 수 배 빠릅니다
```

### 분석/시각화

```
matplotlib       # 결과 그래프
jupyter          # 노트북 분석
```


## 핵심 실험 결과

360회 실험(36시나리오 × 10반복) 기준 주요 수치입니다.

- UW(λ=0.7) 수익 개선: baseline 대비 **+1.86%** (전체 정책 중 1위)
- UW(λ=0.7) 위반율: **15.3%** (Naive 25.8% 대비 41% 감소)
- UW(λ=0.7) regret: **0.00137** (Naive 0.00171 대비 20% 감소)
- UW vs Uniform 우위의 통계적 유의성: t=14.17, p<0.001

주요 패턴으로는, 내생성이 강할수록, 샘플이 적을수록, 이질성이 높을수록 UW의 안전 레이어 효과가 커집니다.


## 관련 논문 구조

이 코드는 다음 논문 구조에 대응됩니다.

| 장 | 내용 | 코드 |
|---|------|------|
| 4장 방법론 | 파이프라인 설계, u(x) 합성, π_safe 유도 | `src/stage0~3` |
| 5장 실험 | 36시나리오 비교, λ sweep, 조건부 효과 | `run_experiment.py`, `results/` |


## 라이선스

연구 목적 프로젝트입니다. M5 데이터 사용 시 Kaggle 대회 규칙을 준수해야 합니다.
