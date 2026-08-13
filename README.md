# Safe Policy Deployment for Continuous Treatment Optimization

> 관찰 데이터에서 추정한 AI 최적 정책이 틀릴 때, 안전하게 배포하는 프레임워크입니다.

---

# Part 1. Quick Start

## What is this?

AI가 추천하는 최적 가격(또는 마진)을 실제로 적용하려면, **추정이 틀릴 리스크**를 관리해야 합니다. 이 프로젝트는 각 상품/개인별로 "얼마나 확실한가"를 측정하고, 불확실할수록 기존 정책에 가까운 안전한 값을 적용하는 방법을 구현합니다.

```
확실한 상품 → AI 추천 가격 그대로 적용  
불확실한 상품 → AI 추천과 기존 가격의 중간값 적용  
매우 불확실한 상품 → 기존 가격 유지  
```

이 동작을 하나의 공식으로 표현하면 다음과 같습니다:

```
π_safe(x) = (1 - w(x)) · π̂*(x) + w(x) · π₀(x),   w(x) = min(λ·u(x), 1)
```

- `π̂*(x)` — 추정 최적 정책 (Stage 1에서 산출)
- `π₀(x)` — Baseline 정책 (현재 운영 중인 고정 마진, 기본값 15%)
- `u(x)` — 개체별 추정 불안정성 (Stage 2에서 산출, 0~1)
- `λ` — 전역 보수성 파라미터 (0 = 보수성 없음, 클수록 보수적)
- `w(x)` — 보간 가중치. λ·u(x)가 1을 넘으면 baseline을 넘어 외삽하게 되므로
  `min(·, 1)`로 클리핑한다 (`src/stage3_deploy.py`).


## 설치

```bash
git clone https://github.com/Budchar/uncertainty-weighted-policy.git
cd uncertainty-weighted-policy
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`xgboost`는 선택 사항입니다 (Stage 1 추정 5~10배 가속, 없으면 sklearn
`GradientBoostingRegressor`로 fallback). `kaggle`은 M5 원본 데이터를
내려받을 때만 필요합니다.

M5 데이터는 없어도 됩니다. 없으면 자동으로 유사 구조의 synthetic 데이터를 생성합니다.


## 첫 실행 (1~2분)

```bash
python run.py
```

전체 파이프라인(데이터 생성 → 추정 → 불확실성 측정 → 정책 비교)을 한 번 실행하고, 5개 정책의 성과를 비교 출력합니다.

출력 형식 (실제 값은 시나리오·시드에 따라 달라집니다):
```
정책                      | 평균수익 | 개선율  | 위반율 | Regret
─────────────────────────┼────────┼───────┼──────┼────────
oracle                    |  0.5234 | +4.10% |  0.0% | 0.00000
uw_lambda0.5              |  0.5112 | +1.80% | 18.1% | 0.00140
naive_optimal             |  0.5098 | +1.29% | 25.8% | 0.00171
```


## 파라미터 조정

```bash
# 데이터 크기와 교란 강도 조정
python run.py --n-samples 5000 --endogeneity 0.5

# DML 대신 GPS 추정기 사용
python run.py --estimator gps

# 이전 실행 결과 캐시 사용 (Stage 2부터 재실행)
python run.py --from-stage 2

# 캐시 무시하고 처음부터
python run.py --force
```


## 실험 실행

```bash
# 빠른 테스트 (5~10분, 24회 실행)
python run_experiment.py --quick

# 축별 실험
python run_experiment.py --axis endogeneity --repeats 5   # 교란 강도별
python run_experiment.py --axis sample_size --repeats 5   # 샘플 크기별
python run_experiment.py --axis heterogeneity --repeats 5 # 이질성별

# 전체 실험 (수시간, 360회 실행)
python run_experiment.py
python run_experiment.py --resume   # 중단 후 이어서 실행

# 상품별 심층 분석 (대표 시나리오 3개)
python run_detailed_analysis.py
```


## M5 데이터 준비 (선택)

M5 데이터 없이도 synthetic covariates로 동작합니다. 실제 M5 데이터를 사용하시려면 아래와 같이 진행하시면 됩니다.

```bash
# Kaggle API 설정 후
python data_loader.py

# 또는 수동으로 m5_data/ 폴더에 배치하시면 됩니다:
#   sales_train_evaluation.csv, sell_prices.csv, calendar.csv
```


## FAQ

**Q: M5 데이터가 꼭 필요한가요?**
A: 아닙니다. M5 데이터가 없으면 유사한 통계적 특성의 synthetic covariates를 자동 생성합니다. M5를 사용하시면 현실적인 상품/매장/시간 구조가 추가됩니다.

**Q: GPU가 필요한가요?**
A: 필수는 아닙니다. XGBoost가 설치되어 있으면 자동으로 사용하고, CUDA GPU가 있으면 GPU 가속까지 적용됩니다. 없으면 sklearn GradientBoostingRegressor로 fallback됩니다.

**Q: 전체 실험이 얼마나 걸리나요?**
A: CPU만 사용 시 수시간, XGBoost+GPU 사용 시 1~2시간 정도입니다. `--quick` 옵션으로 5~10분 내 축소 버전을 먼저 실행하실 수 있습니다.

**Q: 가격 이외의 연속 처치에도 적용 가능한가요?**
A: 프레임워크 자체는 연속 처치(continuous treatment) 일반에 적용 가능합니다. DGP(Stage 0)만 해당 도메인에 맞게 교체하시면 됩니다.


---

# Part 2. 기술 상세

## 프로젝트 구조

```
safe_policy/
├── run.py                          # 단일 파이프라인 실행 (캐시 지원)
├── run_experiment.py               # 체계적 비교 실험 (3축 × 반복)
├── run_detailed_analysis.py        # 상품별 심층 분석 (대표 시나리오)
├── data_loader.py                  # M5 데이터 다운로드 유틸리티
├── requirements.txt
├── LICENSE
│
├── src/                            # 핵심 파이프라인
│   ├── stage0_dgp.py               # Stage 0: 데이터 생성 (M5 기반 반합성)
│   ├── stage1_estimate.py          # Stage 1: Dose-response 추정 (GPS / DML)
│   ├── stage2_quantify.py          # Stage 2: 불확실성 정량화 (Bootstrap + GPS + Conformal)
│   └── stage3_deploy.py            # Stage 3: 정책 구성 및 평가
│
├── configs/
│   └── default_config.py           # DGP 파라미터, 실험 시나리오 설정
│
├── utils/
│   ├── cache_utils.py              # Stage별 캐시 관리
│   └── gpu_utils.py                # XGBoost/GPU 가속 팩토리
│
├── m5_data/                        # M5 원시 데이터 (선택)
├── cache/                          # Stage별 캐시 파일 (자동 생성)
└── results/                        # 실험 결과 (자동 생성)
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

추정의 불확실성을 개인 단위로 정량화합니다. 두 가지 서로 다른 질문에 답합니다:

- **추정이 흔들리는가?** (`u_stat`) — Bootstrap으로 같은 데이터를 여러 번 추정했을 때 최적 가격이 얼마나 달라지는지를 측정합니다
- **해본 적 없는 곳인가?** (`u_pos`) — 그 가격대에서 실제 관찰 데이터가 충분한지를 GPS 기반으로 확인합니다

둘 다 높으면 "추정의 근거 자체가 취약"하므로 상호작용항을 추가하여 더 보수적으로 처리합니다. percentile 정규화로 스케일을 통일한 후 interaction 합성을 적용합니다:

```
u(x) = (stat_p + pos_p + stat_p · pos_p) / 3
```

### Stage 3: Safe Policy Deployment

5개 정책을 비교 평가합니다.

1. **Naive Optimal** — 추정 최적을 그대로 적용합니다 (보수성 없음)
2. **Uniform Conservative** — 모든 개인에 동일한 보수성을 적용합니다
3. **Threshold-based** — 불확실성 임계값 초과 시 baseline으로 전환합니다
4. **Uncertainty-Weighted (제안 방법)** — 불확실성에 비례한 연속적 보간을 수행합니다
5. **Oracle** — ground truth 최적입니다 (성능 상한)

`λ`는 전체적인 보수성 수준을 결정합니다.

- `λ = 0`: 보수성 없음 — 추정 최적을 그대로 적용합니다 (= Naive)
- `λ = 0.5`: 논문의 대표 파라미터. 36개 시나리오 전체에서 Naive 이상의
  수익을 유지하면서 위반율을 최소화하는 값입니다
- `λ ≥ 3`: 매우 보수적 — 거의 기존 정책을 유지합니다

실험에서 `λ = 0.0 ~ 5.0`을 0.1 간격으로 sweep하여 수익-안전성 tradeoff
curve를 그립니다.


## 결과 파일

```
results/
├── experiment_YYYYMMDD_HHMMSS.json    # 전체 원시 결과
├── summary_YYYYMMDD_HHMMSS.csv        # 요약 테이블
├── checkpoint.json                     # Resume용 체크포인트
└── detailed_analysis/
    ├── items_e0.8_n1000_h0.8_*.csv    # 상품별 상세 (어려운 조건)
    ├── items_e0.5_n3000_h0.5_*.csv    # 상품별 상세 (중간 조건)
    └── bands_*.csv                     # u(x) 구간별 분석 테이블
```

### Summary CSV 컬럼

| 컬럼 | 설명 |
|------|------|
| `endogeneity` | 교란 강도 (0.1 ~ 0.8) |
| `n_samples` | 샘플 수 (1K / 3K / 10K) |
| `heterogeneity` | 이질성 (0.2 ~ 0.8) |
| `policy` | 방법 이름 (naive_optimal, uw_lambda0.7, ...) |
| `improvement_over_baseline` | Baseline 대비 수익 개선율(%) |
| `safety_violation_rate` | Baseline보다 나빠지는 비율 |
| `mean_regret` | Oracle 대비 수익 손실 |


---

# Part 3. 연구 내용

## 연구 동기

관찰 데이터 기반 인과추정(causal inference)으로 최적 가격 정책을 도출하는 방법론은 최근 급속히 발전하고 있습니다. 그러나 실무 배포 단계에서 핵심 문제가 남습니다: **추정이 틀릴 경우 기존 정책보다 나쁜 결과를 초래할 수 있습니다.** 특히 관찰 데이터의 미관측 교란(endogeneity), 데이터 부족, 효과 이질성(heterogeneity)이 결합되면 이 위험은 더욱 커집니다.

본 연구는 이 문제를 해결하기 위해 **개인 단위의 불확실성 정량화(uncertainty quantification)**와 **불확실성에 비례한 정책 보간(policy interpolation)**을 결합한 안전한 배포 프레임워크를 제안합니다.


## 핵심 기여

1. **두 축의 epistemic uncertainty 합성**: 통계적 불확실성(bootstrap variance)과 positivity 불확실성(GPS 기반 외삽 위험)을 percentile 정규화 후 상호작용항으로 합성하여, "추정이 흔들리면서 동시에 관찰도 부족한" 복합 위험을 포착합니다.

2. **연속적 정책 보간**: 기존의 이진적 결정(threshold 기반: 적용/불적용)이 아니라, 불확실성에 비례하여 AI 정책과 baseline 사이를 연속적으로 보간합니다. 이를 통해 수익-안전성 tradeoff를 부드럽게 제어할 수 있습니다.

3. **체계적 반합성 실험**: M5 Walmart 데이터의 현실적 covariate 구조 위에 경제학 기반 DGP를 결합하여, 교란 강도 × 샘플 크기 × 이질성의 3축을 체계적으로 변동시킨 360회 실험으로 방법의 강건성을 검증합니다.


## 방법론 요약

### 문제 설정

- 연속 처치(treatment): 마진율 `m ∈ [0.05, 0.35]`
- 결과(outcome): 수익 `Y = demand(x, m) × m`
- 수요 모델: `Q(m) = a(x) · exp(-α(x) · m)` (지수감소, 역U자 revenue curve)
- 최적 마진: `m*(x) = 1/α(x)` (개인별로 상이)

### 설계 근거: 제거된 구성요소

- `u_flat` (dose-response 평탄도): 평탄한 curve는 "어디에 설정해도 비슷하다"는 의미로, epistemic uncertainty가 아니라 오히려 안전 신호에 해당합니다. 이에 제거하였습니다.
- `u_sens` (곡률/민감도): 추정이 맞을 때와 틀릴 때 반대 방향으로 작동하여 보수성 가중치로 부적합합니다. Ablation에서 시나리오별 부호 반전이 확인되어 제거하였습니다.
- `max` 합성: 한 원천이 높으면 나머지를 무시하여 변별력이 감소합니다. interaction으로 대체하였습니다.


## 실험 설계

### 시나리오 (3축)

| 축 | 수준 | 의미 |
|---|------|------|
| 내생성 γ | 0.1, 0.3, 0.5, 0.8 | 미관측 교란의 강도 |
| 샘플 크기 n | 1,000 / 3,000 / 10,000 | 데이터 양 |
| 이질성 h | 0.2, 0.5, 0.8 | 상품 간 가격민감도 차이 |

총 4 × 3 × 3 = 36 시나리오이며, 각 10회 반복하여 **360회 실행**합니다.

### 비교 방법 (5개)

1. **Naive Optimal**: 추정 최적을 보정 없이 적용합니다
2. **Uniform Conservative** (c = 0.05 ~ 0.95, 0.05 간격): 모든 개체에 동일한 보수성을 적용합니다
3. **Threshold-based** (τ = 0.05 ~ 0.95, 0.05 간격): `u(x) > τ`인 개체에 baseline을 적용합니다
4. **Uncertainty-Weighted (제안)**: λ = 0.0 ~ 5.0, 0.1 간격 sweep
5. **Oracle**: ground truth (성능 상한) / **Baseline**: 현행 정책 (하한 기준점)

대표 파라미터는 36개 시나리오 전체에서 Naive 이상의 수익을 유지하면서
위반율을 최소화하는 값으로 선정했습니다 (UW λ=0.5, Uniform c=0.20).
Threshold는 이 조건을 만족하는 τ가 존재하지 않아, 조건을 가장 많이
만족한 τ=0.85를 선정했습니다.

### 평가 지표

- `improvement_over_baseline`: baseline 대비 수익 개선율(%)
- `safety_violation_rate`: baseline보다 수익이 낮아지는 비율
- `mean_regret`: oracle 대비 수익 손실


## 주요 결과 (360회 실험)

대표 파라미터에서의 정책 비교 (360회 평균):

| 정책 | 위반율 | 수익 개선율 | Regret |
|------|--------|------------|--------|
| Oracle | 0.00% | +4.10% | 0.00000 |
| **UW (λ=0.5)** | **18.07%** | **+1.80%** | **0.00140** |
| Uniform (c=0.20) | 19.93% | +1.68% | 0.00147 |
| Threshold (τ=0.85) | 24.72% | +1.33% | 0.00169 |
| Naive | 25.82% | +1.29% | 0.00171 |
| Baseline | 0.00% | 0.00% | 0.00250 |

UW는 Naive 대비 위반율을 25.82% → 18.07%로 30% 감소시키면서 수익 개선율을
+1.29% → +1.80%로 40% 향상시켰고, Regret도 0.00171 → 0.00140으로 감소했습니다.

쌍별 비교 (paired t-test, n=360, 모두 p < 0.001):

| 비교 | Δ위반율 | Δ수익 |
|------|---------|-------|
| UW vs Naive | −7.75%p | +0.51%p |
| UW vs Uniform(c=0.20) | −1.86%p | +0.12%p |
| UW vs Threshold(τ=0.85) | −6.64%p | +0.47%p |

### 조건부 효과 (UW λ=0.5 vs Naive)

| 축 | 조건 | Naive 위반 | UW 위반 | Δ위반 | Δ수익 |
|---|------|-----------|---------|-------|-------|
| γ | 0.1 (n=3K, h=0.5) | 17.2% | 11.6% | −5.6%p | +0.07%p |
| γ | 0.8 (n=3K, h=0.5) | 32.0% | 23.3% | −8.6%p | +0.59%p |
| n | 1K (γ=0.5, h=0.5) | 24.5% | 18.5% | −6.0%p | +0.79%p |
| n | 10K (γ=0.5, h=0.5) | 27.0% | 17.9% | −9.1%p | +0.39%p |

10개 시나리오 모두에서 Δ위반이 음수, Δ수익이 양수입니다. 즉 UW는 모든 조건에서
Naive 대비 위반율을 낮추면서 수익을 유지 또는 향상합니다. 교란이 강할수록
(γ 0.1→0.8) 개선 폭이 확대됩니다.

> 위 수치는 모두 `results/summary_20260416_200920.csv`(360런)에서 재계산해
> 검증한 값입니다.


## 재현 가이드

```bash
# 빠른 검증 (5~10분, 축소 버전 24회)
python run_experiment.py --quick

# 전체 재현 (수시간, 360회)
python run_experiment.py

# 심층 분석 재현 (대표 시나리오 3개)
python run_detailed_analysis.py
```

결과는 `results/` 디렉토리에 JSON(원시)과 CSV(요약)로 저장됩니다. 중단 시 `--resume`으로 이어서 실행하실 수 있습니다.


## 논문과의 대응

이 저장소는 다음 논문의 실험 코드입니다.

> 추정 불안정성 가중 보간을 통한 연속 처리 정책의 안전한 배포
> (Uncertainty-Weighted Policy Interpolation for Safe Deployment in
> Continuous Treatment Settings)

| 논문 절 | 내용 | 코드 |
|---|------|------|
| 2.2 인과추정 파이프라인 | DML, 5-fold cross-fitting, XGBoost nuisance | `src/stage1_estimate.py` |
| 2.3 추정 불안정성 정량화 | 식(1) `u(x)` 합성 (bootstrap B=20 + GPS positivity) | `src/stage2_quantify.py` |
| 2.4 불확실성 가중 정책 보간 | 식(2) `π_safe` | `src/stage3_deploy.py` |
| 3.1 실험 설계 | 36시나리오 × 10반복 = 360런, λ/c/τ sweep | `run_experiment.py` (`FULL_CONFIG`) |
| 3.2~3.4 결과 | 표1·표2, 프론티어 비교 | `results/summary_20260416_200920.csv` |


## 한계 및 향후 연구

1. **In-sample 평가**: 현재 모든 평가가 학습 데이터 내에서 이루어집니다. Out-of-sample 검증이 필요합니다.
2. **λ의 시나리오 의존성**: 36개 시나리오 각각에서 "Naive 이상의 수익을 유지하며
   위반율을 최소화하는 λ"를 구하면 0.5 ~ 5.0에 넓게 분산되며 지배적인 값이
   없습니다 (최빈값 λ=1.1과 λ=5.0이 각 17%). 논문의 λ=0.5는 전 시나리오에서
   동시에 조건을 만족하는 보수적 선택이며, 시나리오를 알 수 있다면 더 높은 λ가
   유리한 경우가 많습니다. Validation 기반 자동 선택 알고리즘이 필요합니다.
3. **반합성 환경의 한계**: 실제 A/B 테스트 환경에서의 검증이 남아 있습니다.
4. **Stage 1 추정기**: 현재 GBR/XGBoost 기반이며, NN-DML 등 flexible 추정기와의 비교가 가능합니다.


## 참고 문헌

- Chernozhukov, V. et al. (2018). Double/debiased machine learning for treatment and causal parameters. *The Econometrics Journal*.
- Hirano, K. & Imbens, G. W. (2004). The propensity score with continuous treatments. *Applied Bayesian Modeling and Causal Inference from Incomplete-Data Perspectives*.
- Vovk, V. et al. (2005). Algorithmic Learning in a Random World. (Conformal Prediction)
- Romano, Y. et al. (2019). Conformalized Quantile Regression. *NeurIPS*.


## 라이선스

이 저장소의 코드는 MIT License로 배포됩니다 ([LICENSE](LICENSE) 참조).

M5 원본 데이터는 재배포하지 않습니다. 데이터를 사용하시려면 Kaggle에서 직접
내려받으시고, Kaggle Competition Rules를 준수해 주시기 바랍니다. 데이터 없이
synthetic covariates만으로도 전체 파이프라인이 동작합니다.
