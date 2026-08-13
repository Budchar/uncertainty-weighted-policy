# 체계적 비교 실험 실행 가이드

논문 3장(실험)의 360회 실험을 재현하기 위한 실행 절차입니다.
프로젝트 개요와 방법론은 [README.md](README.md)를 참조하세요.

## 실행 방법

### 1단계: 빠른 검증 (5~10분)
먼저 파이프라인이 정상 동작하는지 확인합니다.
```bash
python run_experiment.py --quick
```
축소 버전: 2×2×2 시나리오, 3회 반복 = 24회 실행

### 2단계: 단일 축 실험 (10~30분)
특정 축만 먼저 돌려서 패턴을 확인합니다.
```bash
python run_experiment.py --axis endogeneity   --repeats 5
python run_experiment.py --axis sample_size   --repeats 5
python run_experiment.py --axis heterogeneity --repeats 5
```

### 3단계: 전체 실험 (수시간)
```bash
python run_experiment.py
```
4×3×3 = 36 시나리오 × 10회 반복 = **360회 실행** (논문 표1·표2의 원본)

중단 후 이어서 실행:
```bash
python run_experiment.py --resume
```

## 실험 그리드

`run_experiment.py`의 `FULL_CONFIG`에서 정의합니다.

| 항목 | 값 |
|---|---|
| 교란 강도 γ | 0.1, 0.3, 0.5, 0.8 |
| 표본 크기 n | 1,000 / 3,000 / 10,000 |
| 이질성 h | 0.2, 0.5, 0.8 |
| λ (UW) | 0.0 ~ 5.0, 0.1 간격 (51개) |
| c (Uniform) | 0.05 ~ 0.95, 0.05 간격 (19개) |
| τ (Threshold) | 0.05 ~ 0.95, 0.05 간격 (19개) |
| 반복 | 10회 |
| Bootstrap B | 20 |
| 추정기 | DML (5-fold cross-fitting, XGBoost nuisance) |

## 결과 파일

```
results/
├── experiment_YYYYMMDD_HHMMSS.json   # 전체 원시 결과
├── summary_YYYYMMDD_HHMMSS.csv       # 요약 테이블 (분석용)
└── checkpoint.json                    # 체크포인트 (resume용)
```

### summary CSV 컬럼
- `endogeneity`, `n_samples`, `heterogeneity`: 시나리오 설정
- `repeat`, `seed`: 반복 정보
- `policy`: 방법 이름 (`naive_optimal`, `uw_lambda0.5`, `uniform_c0.2`, ...)
- `mean_revenue`: 평균 수익
- `safety_violation_rate`: 위반율 (baseline 대비 수익이 하락한 개체 비율)
- `mean_regret`: oracle 대비 손실
- `improvement_over_baseline`: baseline 대비 개선율(%)
- `s1_optimal_t_mae`: Stage 1 추정 오차

## 관측된 핵심 결과

아래는 `results/summary_20260416_200920.csv`(360런)에서 재계산한 값입니다.
UW는 λ=0.5, 비교 대상은 Naive입니다.

1. **교란이 강할수록 UW의 개선 폭이 커진다** (n=3K, h=0.5 기준)
   - γ=0.1: Δ위반 −5.6%p, Δ수익 +0.07%p
   - γ=0.8: Δ위반 −8.6%p, Δ수익 +0.59%p
   - 논문의 핵심 서사에 해당합니다.

2. **표본 크기 축은 지표별로 방향이 다르다** (γ=0.5, h=0.5 기준)
   - 위반율 감소 폭은 n이 **클수록** 커집니다 (n=1K −6.0%p → n=10K −9.1%p).
   - 반면 수익 이득은 n이 **작을수록** 큽니다 (n=1K +0.79%p → n=10K +0.39%p).
   - 즉 "데이터가 적을수록 안전 레이어가 중요하다"는 서술은 수익 측면에서만
     성립하며, 위반율 측면에서는 반대 방향입니다.

3. **이질성이 낮을수록 위반율 감소 폭이 크다** (γ=0.5, n=3K 기준)
   - h=0.2: Δ위반 −9.2%p, Δ수익 +0.49%p
   - h=0.8: Δ위반 −6.9%p, Δ수익 +0.28%p
   - 사전 예상(이질성이 높을수록 개체별 측정의 가치가 크다)과는 반대 방향이므로
     해석에 주의가 필요합니다.

4. **λ tradeoff curve**
   - λ를 올리면 위반율↓, 수익↓의 단조 트레이드오프가 나타납니다.
   - 시나리오별 최적 λ는 0.5~5.0에 넓게 분산되어 지배적인 값이 없습니다.
     논문의 λ=0.5는 36개 시나리오 전체에서 동시에 "Naive 이상 수익 유지"를
     만족하는 보수적 선택입니다.
