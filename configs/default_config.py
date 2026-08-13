"""
DGP 설정 파일
=============
M5 로더와 반합성 DGP의 확정된 파라미터를 모아둔 곳.

실험 시나리오 그리드(교란 강도 × 표본 크기 × 이질성, λ/c/τ sweep)는
run_experiment.py의 FULL_CONFIG / QUICK_CONFIG에서 정의한다.

핵심 설계 원칙:
- M5의 현실적 covariate 구조 + 지수감소 수요 모델 기반 semi-synthetic DGP
- 내생성: 가격/판매에 비슷한 스케일로 영향 (이전 실험에서 발견된 불균형 수정)
- dose-response: 역U자형(단봉, unimodal) — 경제학적 근거 있음
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class M5Config:
    """M5 데이터에서 추출할 covariate 설정"""
    data_dir: str = "./m5_data"
    
    # M5에서 추출할 핵심 covariates (10~20개로 제한 — 초점을 유지하기 위해)
    # 상품 계층: dept_id, cat_id
    # 매장: store_id, state_id
    # 시간: weekday, month, week_of_year
    # 이벤트: event_type_1, snap_{CA,TX,WI}
    # 파생: rolling_mean_sales, sales_volatility
    
    # 사용할 카테고리 (전체 쓰면 너무 큼)
    categories: List[str] = field(default_factory=lambda: ["FOODS", "HOUSEHOLD", "HOBBIES"])
    
    # 샘플링: 상품 수 제한
    n_items_per_category: int = 100  # 카테고리당 100개 → 총 300개 상품
    
    # 시간 범위: 마지막 52주 (1년)
    n_weeks: int = 52


@dataclass
class DGPConfig:
    """Semi-synthetic Data Generating Process 설정
    
    경제학적 근거:
    - 지수감소 수요: Q(m) = a · exp(-α · m), Revenue = a · m · exp(-α · m)
    - 최적 마진 m* = 1/α, Revenue curve는 역U자형 (단봉, unimodal)
    - 이 구조에서 interpolation이 안전한 이유: 두 정책 사이의 어떤 값도
      양 끝보다 나쁠 가능성이 낮음
    """
    
    # --- Treatment (마진율) 설정 ---
    # 마진율 범위: 5% ~ 35%
    margin_min: float = 0.05
    margin_max: float = 0.35
    
    # Baseline 정책: 모든 상품에 일률적 15% 마진
    baseline_margin: float = 0.15
    
    # --- Demand function 파라미터 ---
    # 지수감소 수요 모델: Q(m) = base_demand(x) × exp(-α(x) × m)
    # Revenue = Q × m = base_demand × m × exp(-α × m)
    # 최적 마진: m* = 1/α  (역U자형 revenue curve의 꼭대기)
    #
    # 왜 isoelastic이 아닌가:
    #   isoelastic에서 t* = 1/(ε-1)은 마진율 5~35% 범위 밖에 위치하여
    #   dose-response가 단조증가가 되어 역U자가 나타나지 않음.
    #   지수감소 모델은 m* = 1/α로, α 조절만으로 최적점을 범위 안에 배치 가능.
    
    # 카테고리별 가격민감도(α) 범위
    # α 높음 = 가격 민감(최적 마진 낮음), α 낮음 = 가격 둔감(최적 마진 높음)
    # FOODS: m* ≈ 10~17% (가격 민감), HOBBIES: m* ≈ 20~33% (취미는 덜 민감)
    alpha_by_category: Dict[str, tuple] = field(default_factory=lambda: {
        "FOODS": (6.0, 10.0),       # m* = 10~17%, 가격 민감
        "HOUSEHOLD": (4.0, 7.0),    # m* = 14~25%, 중간
        "HOBBIES": (3.0, 5.0),      # m* = 20~33%, 가격 둔감
    })
    
    # 이질성 강도 (covariates에 의해 α가 달라지는 정도)
    # 0이면 카테고리 내 균일, 1이면 covariates에 크게 의존
    heterogeneity_strength: float = 0.5
    
    # --- 내생성 (Confounding) 설정 ---
    # 핵심: 가격과 판매에 비슷한 스케일로 영향 줘야 함
    # (이전 실험에서 1.0 vs 0.05로 20배 차이 나서 추정이 완전 틀어졌음)
    endogeneity_strength: float = 0.3  # 0.0 ~ 1.0
    
    # 미관측 수요 신호의 영향 스케일
    # 가격에 미치는 영향: endogeneity_strength * confound_scale
    # 판매에 미치는 영향: endogeneity_strength * confound_scale
    confound_scale: float = 0.05  # 5% 수준
    
    # --- 노이즈 ---
    price_noise_std: float = 0.03    # 가격 결정의 랜덤 노이즈 (±3%)
    outcome_noise_std: float = 0.05  # 판매량의 랜덤 노이즈
    
    # --- 데이터 크기 ---
    n_samples: int = 10000  # 기본 샘플 수
    
    # 랜덤 시드
    random_seed: int = 42
