# Safe Policy Deployment for Continuous Treatment Optimization
# 연속 처치 최적화에서 안전한 정책 배포 프레임워크

## 연구 개요

**핵심 질문**: "관찰 데이터에서 추정한 최적 가격 정책이 틀릴 때, 어떻게 안전하게 배포할 수 있는가?"

**핵심 공식**: `π_safe(x) = (1 - λ·u(x)) · π*(x) + λ·u(x) · π₀(x)`
- π*: AI 최적 정책, π₀: baseline 정책, u(x): 불확실성 (0~1)

## 프로젝트 구조

```
safe_policy/
├── configs/
│   └── default_config.py      # 실험 설정 (DGP 파라미터, 실험 시나리오)
├── src/
│   ├── stage0_dgp.py          # ✅ Stage 0: Semi-synthetic 데이터 생성
│   ├── stage1_estimate.py     # 🔲 Stage 1: Dose-response curve 추정 (DML 기반)
│   ├── stage2_quantify.py     # 🔲 Stage 2: Conformal prediction 불확실성
│   ├── stage3_deploy.py       # 🔲 Stage 3: Uncertainty-weighted policy
│   └── stage4_experiment.py   # 🔲 Stage 4: 비교 실험 & 결과 분석
└── notebooks/
    └── (시각화/분석 노트북)
```

## 구현 현황

### ✅ Stage 0: Semi-Synthetic DGP (완료)
- M5 covariates 추출 (또는 synthetic fallback)
- Isoelastic demand 기반 treatment/outcome 생성
- 내생성 강도 조절 가능
- Ground truth (Oracle) 계산 포함
- 진단 유틸리티 포함

### 🔲 Stage 1: Estimate (다음)
- GPS (Generalized Propensity Score) 추정
- Dose-response curve 추정 (DML 또는 GPS식)
- Bootstrap confidence intervals

### 🔲 Stage 2: Quantify
- Conformal prediction으로 개인별 불확실성 구간
- Positivity violation 진단
- 불확실성 합성 (통계적 + positivity + sensitivity)

### 🔲 Stage 3: Deploy
- Uncertainty-weighted policy interpolation 구현
- Baseline 비교 방법 구현 (uniform, threshold)
- Regret 및 safety 지표 계산

### 🔲 Stage 4: Experiments
- 3축 실험 (교란 강도 × 샘플 크기 × 이질성)
- 5개 방법 비교
- 결과 시각화 및 분석

## 실행 방법

```bash
# Stage 0 테스트
cd safe_policy
python src/stage0_dgp.py

# M5 데이터 사용 시: m5_data/ 폴더에 M5 파일 배치 후 실행
```

## 의존성

```
numpy
pandas
scikit-learn
```

Stage 1부터 추가 필요:
```
econml         # DML 등
matplotlib     # 시각화
```
