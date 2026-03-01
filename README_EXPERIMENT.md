# 체계적 비교 실험 실행 가이드

## 파일 배치
`run_experiment.py`를 기존 `safe_policy/` 프로젝트 루트에 복사하세요.

```
safe_policy/
├── run_experiment.py   ← 이 파일 추가
├── run.py              ← 기존 단일 파이프라인
├── configs/
├── src/
│   ├── stage0_dgp.py
│   ├── stage1_estimate.py
│   ├── stage2_quantify.py
│   └── stage3_deploy.py
├── m5_data/
├── cache/
└── results/            ← 실험 결과 자동 생성
```

## 실행 방법

### 1단계: 빠른 테스트 (5~10분)
먼저 코드가 정상 동작하는지 확인:
```bash
python run_experiment.py --quick
```
축소 버전: 2×2×2 시나리오, 3회 반복 = 24회 실행

### 2단계: 단일 축 실험 (10~30분)
특정 축만 먼저 돌려서 패턴 확인:
```bash
# 내생성 강도별 비교 (핵심 서사: 내생성 강할수록 안전 레이어 효과 큼)
python run_experiment.py --axis endogeneity --repeats 5

# 샘플 크기별 비교 (서사: 데이터 적을수록 안전 레이어 중요)
python run_experiment.py --axis sample_size --repeats 5

# 이질성별 비교 (서사: 개인차 클수록 개인별 불확실성 측정 중요)
python run_experiment.py --axis heterogeneity --repeats 5
```

### 3단계: 전체 실험 (수시간)
```bash
python run_experiment.py
```
4×3×3 시나리오 × 10회 반복 = 360회 실행

중단 후 이어서 실행:
```bash
python run_experiment.py --resume
```

## 결과 파일

```
results/
├── experiment_20260216_143022.json   # 전체 원시 결과
├── summary_20260216_143022.csv       # 요약 테이블 (분석용)
└── checkpoint.json                   # 체크포인트 (resume용)
```

### summary CSV 컬럼:
- endogeneity, n_samples, heterogeneity: 시나리오 설정
- repeat, seed: 반복 정보
- policy: 방법 이름
- mean_revenue: 평균 수익
- safety_violation_rate: 위반율
- mean_regret: oracle 대비 손실
- improvement_over_baseline: baseline 대비 개선율(%)
- s1_optimal_t_mae: Stage 1 추정 오차

## 기대하는 핵심 발견

1. **내생성 강할수록 안전 레이어 효과 큼**
   - endo=0.1: Naive도 위반 낮음 → UW 효과 작음
   - endo=0.8: Naive 위반 높음 → UW가 크게 줄여줌

2. **샘플 작을수록 안전 레이어 중요**
   - n=10000: 추정 자체가 정확 → UW 효과 작음
   - n=1000: 추정 부정확 → UW 효과 큼

3. **이질성 높을수록 개인별 불확실성 측정 가치 증가**
   - hetero=0.2: 상품 간 차이 작음 → uniform conservative와 UW 비슷
   - hetero=0.8: 상품 간 차이 큼 → UW가 uniform 대비 우월

4. **λ tradeoff curve**
   - λ 올리면: 위반율↓ 수익↓
   - 최적 균형점이 시나리오에 따라 달라짐