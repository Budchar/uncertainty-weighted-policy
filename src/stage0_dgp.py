"""
Stage 0: Semi-Synthetic Data Generating Process (DGP)
=====================================================

역할: M5 데이터의 현실적 covariate 구조 위에 경제학 기반 인과 메커니즘을 씌워서
      ground truth를 아는 실험 환경을 구성한다.

설계 원칙:
1. Covariates (X): M5에서 추출 — 상품/매장/시간/이벤트의 자연스러운 상관 구조 유지
2. Treatment (T): 마진율 — GPS(Generalized Propensity Score) 기반 할당
3. Outcome (Y): 수익 = 판매량 × 마진 — 지수감소 수요 모델(exponential decay demand)
4. Confounding (U): 미관측 수요 신호 — 가격/판매 양쪽에 비슷한 스케일로 영향

수요 모델:
- 지수감소 수요: Q(m) = a · exp(-α · m)
- Revenue = Q × margin = a · m · exp(-α · m) → 역U자형 (단봉, unimodal)
- 최적 마진: m* = 1/α
- Isoelastic 모델(Q=a·p^(-ε))은 최적점이 마진 범위 밖에 위치하여 채택하지 않음
  (상세 근거는 SemiSyntheticDGP 클래스 docstring 참조)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import asdict

import sys
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from configs.default_config import DGPConfig, M5Config


# =============================================================================
# Part 1: M5 Covariate Extraction
# =============================================================================

def load_m5_covariates(config: Optional[M5Config] = None) -> pd.DataFrame:
    """M5 데이터에서 핵심 covariates를 추출한다.
    
    M5가 없으면 synthetic covariates를 생성 (개발/테스트용).
    
    Returns:
        DataFrame with columns:
        - item_id: 상품 식별자
        - cat_id: 카테고리 (FOODS, HOUSEHOLD, HOBBIES)
        - dept_id: 부서
        - store_id: 매장
        - state_id: 주
        - weekday: 요일 (0-6)
        - month: 월 (1-12)
        - is_event: 이벤트 여부
        - snap: SNAP 수급 여부
        - base_sales: 기본 판매량 (M5의 실제 평균 판매)
        - sales_volatility: 판매 변동성
    """
    if config is None:
        config = M5Config()
    
    data_path = Path(config.data_dir)
    
    # M5 파일 존재 여부 확인
    sales_file = data_path / "sales_train_evaluation.csv"
    prices_file = data_path / "sell_prices.csv"
    calendar_file = data_path / "calendar.csv"
    
    if all(f.exists() for f in [sales_file, prices_file, calendar_file]):
        return _extract_from_m5(config)
    else:
        print("⚠ M5 데이터가 없습니다. Synthetic covariates를 생성합니다.")
        print(f"  (M5 데이터를 {config.data_dir}에 넣으면 실제 데이터를 사용합니다)")
        return _generate_synthetic_covariates(config)


def _extract_from_m5(config: M5Config) -> pd.DataFrame:
    """M5 실제 데이터에서 covariates 추출"""
    data_path = Path(config.data_dir)
    
    # 1. 판매 데이터 로드 (상품 메타데이터 + 판매량)
    print("M5 판매 데이터 로딩...")
    sales = pd.read_csv(data_path / "sales_train_evaluation.csv")
    
    # 메타데이터 컬럼
    meta_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    # 판매 컬럼 (마지막 n_weeks * 7일)
    day_cols = [c for c in sales.columns if c.startswith('d_')]
    recent_days = day_cols[-(config.n_weeks * 7):]  # 마지막 1년
    
    # 2. 카테고리별 샘플링
    sampled_items = []
    for cat in config.categories:
        cat_items = sales[sales['cat_id'] == cat]
        n_sample = min(config.n_items_per_category, len(cat_items))
        sampled = cat_items.sample(n=n_sample, random_state=42)
        sampled_items.append(sampled)
    
    sales_sampled = pd.concat(sampled_items, ignore_index=True)
    
    # 3. 상품별 통계 계산
    sales_values = sales_sampled[recent_days].values
    
    records = []
    for idx, row in sales_sampled.iterrows():
        item_sales = sales_values[len(records)]  # 해당 상품의 일별 판매
        
        records.append({
            'item_id': row['item_id'],
            'cat_id': row['cat_id'],
            'dept_id': row['dept_id'],
            'store_id': row['store_id'],
            'state_id': row['state_id'],
            'base_sales': float(np.mean(item_sales)),
            'sales_volatility': float(np.std(item_sales)),
            'sales_median': float(np.median(item_sales)),
            'zero_sales_ratio': float(np.mean(item_sales == 0)),
        })
    
    items_df = pd.DataFrame(records)
    
    # 4. 캘린더 정보 로드
    calendar = pd.read_csv(data_path / "calendar.csv")
    
    # 주간 단위로 집계 (week_id 기준)
    weekly_calendar = (
        calendar
        .groupby('wm_yr_wk')
        .agg({
            'weekday': 'first',
            'month': 'first',
            'event_name_1': lambda x: int(x.notna().any()),
            'snap_CA': 'max',
            'snap_TX': 'max',
            'snap_WI': 'max',
        })
        .reset_index()
    )
    weekly_calendar.rename(columns={
        'event_name_1': 'is_event',
    }, inplace=True)
    
    # 마지막 n_weeks만
    weekly_calendar = weekly_calendar.tail(config.n_weeks).reset_index(drop=True)
    weekly_calendar['week'] = range(config.n_weeks)
    
    # 5. 상품 × 주 의 cross join
    items_df['_key'] = 1
    weekly_calendar['_key'] = 1
    covariates = items_df.merge(weekly_calendar, on='_key').drop('_key', axis=1)
    
    # SNAP 변수: 매장 주(state)에 맞는 snap 적용
    covariates['snap'] = 0
    for state in ['CA', 'TX', 'WI']:
        mask = covariates['state_id'] == state
        covariates.loc[mask, 'snap'] = covariates.loc[mask, f'snap_{state}']
    covariates.drop(columns=['snap_CA', 'snap_TX', 'snap_WI'], inplace=True)
    
    # 6. 범주형 → 수치형 인코딩
    covariates = _encode_categoricals(covariates)
    
    print(f"✓ M5 covariates 추출 완료: {len(covariates)} rows "
          f"({len(items_df)} items × {config.n_weeks} weeks)")
    
    return covariates


def _generate_synthetic_covariates(config: M5Config) -> pd.DataFrame:
    """M5가 없을 때 유사한 구조의 합성 covariates 생성 (개발/테스트용)"""
    np.random.seed(42)
    
    records = []
    item_counter = 0
    
    stores = ['CA_1', 'CA_2', 'CA_3', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
    
    for cat in config.categories:
        for i in range(config.n_items_per_category):
            item_id = f"{cat}_{i:03d}"
            store = np.random.choice(stores)
            state = store.split('_')[0]
            
            # 카테고리별 기본 판매량 차이
            base_sales_mean = {"FOODS": 5.0, "HOUSEHOLD": 2.5, "HOBBIES": 1.5}[cat]
            base_sales = max(0.1, np.random.lognormal(
                np.log(base_sales_mean), 0.5
            ))
            
            for week in range(config.n_weeks):
                month = (week // 4) % 12 + 1
                weekday = week % 7
                is_event = int(np.random.random() < 0.1)  # 10% 확률로 이벤트
                snap = int(np.random.random() < 0.15)      # 15% 확률로 SNAP
                
                records.append({
                    'item_id': item_id,
                    'cat_id': cat,
                    'dept_id': f"{cat}_dept",
                    'store_id': store,
                    'state_id': state,
                    'base_sales': base_sales,
                    'sales_volatility': base_sales * np.random.uniform(0.3, 0.8),
                    'sales_median': base_sales * np.random.uniform(0.7, 1.0),
                    'zero_sales_ratio': np.random.uniform(0.0, 0.3),
                    'wm_yr_wk': 11100 + week,
                    'weekday': weekday,
                    'month': month,
                    'week': week,
                    'is_event': is_event,
                    'snap': snap,
                })
    
    covariates = pd.DataFrame(records)
    covariates = _encode_categoricals(covariates)
    
    print(f"✓ Synthetic covariates 생성 완료: {len(covariates)} rows")
    return covariates


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """범주형 변수를 수치형으로 인코딩"""
    df = df.copy()
    
    # cat_id → 수치형
    cat_map = {"FOODS": 0, "HOUSEHOLD": 1, "HOBBIES": 2}
    df['cat_encoded'] = df['cat_id'].map(cat_map)
    
    # state_id → 수치형
    state_map = {"CA": 0, "TX": 1, "WI": 2}
    df['state_encoded'] = df['state_id'].map(state_map)
    
    # store_id → 수치형 (label encoding)
    stores = sorted(df['store_id'].unique())
    store_map = {s: i for i, s in enumerate(stores)}
    df['store_encoded'] = df['store_id'].map(store_map)
    
    # weekday 처리: M5 calendar에서는 문자열("Saturday" 등)로 되어 있음
    weekday_sample = df['weekday'].dropna().iloc[0] if len(df) > 0 else 0
    if isinstance(weekday_sample, str):
        weekday_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6,
        }
        df['weekday'] = df['weekday'].map(weekday_map).fillna(0).astype(float)
    else:
        df['weekday'] = pd.to_numeric(df['weekday'], errors='coerce').fillna(0).astype(float)
    
    # month 처리: 문자열일 수 있음
    month_sample = df['month'].dropna().iloc[0] if len(df) > 0 else 1
    if isinstance(month_sample, str):
        df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(1).astype(float)
    else:
        df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(1).astype(float)
    
    # 시간 특성: 주기적 인코딩
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    
    return df


# =============================================================================
# Part 2: Semi-Synthetic DGP
# =============================================================================

class SemiSyntheticDGP:
    """Semi-synthetic 데이터 생성기
    
    M5 covariates 위에 지수감소 수요 모델 기반 인과 메커니즘을 씌운다.
    
    인과 구조 (DAG):
        X (covariates) → T (margin)
        X → Y (revenue)
        U (unobserved demand) → T
        U → Y
        T → Y  (이것이 우리가 추정하려는 인과 효과)
    
    핵심 함수:
        demand(x, m) = base_demand(x) × exp(-α(x) × m)
        revenue(x, m) = demand(x, m) × m
        optimal margin: m*(x) = 1/α(x)
        
    여기서 m은 마진율 (0.05 ~ 0.35), α(x)는 개인별 가격민감도
    
    왜 지수감소 모델인가:
        - Isoelastic 모델(Q=a·p^(-ε))에서 최적 마진 t*=1/(ε-1)은 
          현실적 마진 범위(5~35%) 밖에 위치하여 단조증가 → 역U자 불가능
        - 지수감소 모델은 m*=1/α로, 최적점이 자연스럽게 범위 안에 위치
        - 선형 수요 모델(Q=a-bp)과 유사한 역U자 특성을 가지면서도
          수요가 음수가 되지 않는 장점
    """
    
    def __init__(self, config: Optional[DGPConfig] = None):
        self.config = config or DGPConfig()
        self.rng = np.random.RandomState(self.config.random_seed)
    
    def generate(
        self,
        covariates: pd.DataFrame,
        n_samples: Optional[int] = None,
        endogeneity_strength: Optional[float] = None,
        heterogeneity_strength: Optional[float] = None,
    ) -> pd.DataFrame:
        """Semi-synthetic 데이터 생성
        
        Args:
            covariates: M5에서 추출한 covariates DataFrame
            n_samples: 샘플 수 (None이면 covariates 전체 사용)
            endogeneity_strength: 내생성 강도 override
            heterogeneity_strength: 이질성 강도 override
            
        Returns:
            DataFrame with:
            - 모든 covariate 컬럼
            - treatment: 마진율 (연속, 0.05~0.35)
            - outcome: 수익 (revenue)
            - true_alpha: ground truth 가격민감도
            - true_optimal_margin: ground truth 최적 마진 (= 1/α)
            - true_optimal_revenue: ground truth 최적 수익
            - unobserved_confounder: 미관측 교란 변수 (검증용)
        """
        cfg = self.config
        endo = endogeneity_strength if endogeneity_strength is not None else cfg.endogeneity_strength
        hetero = heterogeneity_strength if heterogeneity_strength is not None else cfg.heterogeneity_strength
        
        # 샘플링
        if n_samples is not None and n_samples < len(covariates):
            data = covariates.sample(n=n_samples, random_state=self.rng).reset_index(drop=True)
        else:
            data = covariates.copy().reset_index(drop=True)
            n_samples = len(data)
        
        # Step 1: 개인별 가격민감도(α) 할당
        alphas = self._assign_alphas(data, hetero)
        
        # Step 2: 기본 수요 (base demand) 설정
        base_demands = self._compute_base_demand(data)
        
        # Step 3: 미관측 교란 변수 생성
        unobserved = self.rng.normal(0, 1, size=n_samples)
        
        # Step 4: Treatment (마진율) 할당 — GPS 기반 + 내생성
        treatments = self._assign_treatment(data, unobserved, endo)
        
        # Step 5: Outcome (수익) 생성
        outcomes, demands = self._generate_outcome(
            base_demands, alphas, treatments, unobserved, endo
        )
        
        # Step 6: Ground truth 계산 (Oracle용)
        optimal_margins, optimal_revenues = self._compute_oracle(
            base_demands, alphas
        )
        
        # 결과 조합
        data['treatment'] = treatments
        data['outcome'] = outcomes
        data['demand'] = demands
        data['true_alpha'] = alphas
        data['true_optimal_margin'] = optimal_margins
        data['true_optimal_revenue'] = optimal_revenues
        data['unobserved_confounder'] = unobserved
        data['base_demand'] = base_demands
        
        # Baseline 정책의 수익 계산 (safety 평가용)
        baseline_revenues, _ = self._compute_revenue(
            base_demands, alphas,
            np.full(n_samples, cfg.baseline_margin),
            np.zeros(n_samples),  # 교란 없는 기대값
            0.0
        )
        data['baseline_revenue'] = baseline_revenues
        
        return data
    
    def _assign_alphas(
        self, data: pd.DataFrame, heterogeneity: float
    ) -> np.ndarray:
        """개인별 가격민감도(α) 할당
        
        α가 클수록 가격에 민감 → 최적 마진(1/α)이 낮음
        카테고리별 기본 범위 + covariates에 의한 이질성
        """
        n = len(data)
        alphas = np.zeros(n)
        
        for cat, (alpha_low, alpha_high) in self.config.alpha_by_category.items():
            mask = data['cat_id'] == cat
            n_cat = mask.sum()
            
            if n_cat == 0:
                continue
            
            # 카테고리 기본 α
            base_alpha = (alpha_low + alpha_high) / 2
            alpha_range = (alpha_high - alpha_low) / 2
            
            # Covariates에 의한 이질성
            cat_data = data.loc[mask]
            
            if 'base_sales' in cat_data.columns:
                # base_sales 높은 상품 → 더 가격 민감 (대체재 많음) → α 높음
                sales_vals = cat_data['base_sales'].values
                sales_norm = (sales_vals - sales_vals.mean()) / (sales_vals.std() + 1e-8)
                sales_norm = np.clip(sales_norm, -2, 2) / 2  # [-1, 1]
            else:
                sales_norm = np.zeros(n_cat)
            
            # 이벤트/SNAP이면 가격민감도 증가 (프로모션에 민감)
            event_effect = np.zeros(n_cat)
            if 'is_event' in cat_data.columns:
                event_effect += cat_data['is_event'].values * 0.3
            if 'snap' in cat_data.columns:
                event_effect += cat_data['snap'].values * 0.2
            
            # 최종 α = 기본 + 이질성(covariates 기반) + 노이즈
            individual_alpha = (
                base_alpha
                + heterogeneity * alpha_range * sales_norm
                + heterogeneity * event_effect
                + self.rng.normal(0, 0.3, size=n_cat)
            )
            
            # 범위 제한 (α 최소 2.0, 최대 15.0)
            # α=2 → m*=0.50, α=15 → m*=0.067
            individual_alpha = np.clip(individual_alpha, 2.0, 15.0)
            
            alphas[mask] = individual_alpha
        
        return alphas
    
    def _compute_base_demand(self, data: pd.DataFrame) -> np.ndarray:
        """기본 수요량 계산 (마진율 적용 전)"""
        if 'base_sales' in data.columns:
            # M5의 실제 판매량 기반
            base = data['base_sales'].values.copy()
            
            # 시간 효과 (계절성)
            if 'month_sin' in data.columns:
                seasonality = (
                    0.1 * data['month_sin'].values
                    + 0.05 * data['month_cos'].values
                )
                base = base * (1 + seasonality)
            
            # 이벤트 효과
            if 'is_event' in data.columns:
                base = base * (1 + 0.2 * data['is_event'].values)
            
            # 최소값 보장
            base = np.maximum(base, 0.1)
        else:
            # Fallback
            base = self.rng.lognormal(1.0, 0.5, size=len(data))
        
        return base
    
    def _assign_treatment(
        self,
        data: pd.DataFrame,
        unobserved: np.ndarray,
        endogeneity: float,
    ) -> np.ndarray:
        """Treatment (마진율) 할당
        
        마진율 = 기본 마진 + covariates 효과 + 내생성 + 노이즈
        
        내생성 메커니즘:
        - 수요가 높을 것 같으면 마진을 올림 (기업의 합리적 행동)
        - unobserved demand signal이 양수 → 마진 인상
        """
        cfg = self.config
        n = len(data)
        
        # 기본 마진 (카테고리별)
        base_margin = np.full(n, cfg.baseline_margin)
        
        # Covariates에 의한 관찰 가능한 마진 변동
        # (예: 인기 매장은 마진을 약간 높게)
        covariate_effect = np.zeros(n)
        if 'base_sales' in data.columns:
            sales_norm = (data['base_sales'].values - data['base_sales'].mean()) / (data['base_sales'].std() + 1e-8)
            covariate_effect += 0.02 * np.clip(sales_norm, -2, 2)
        
        # 내생성: 미관측 수요 신호 → 마진 결정
        # (핵심: 가격에 미치는 스케일 = confound_scale)
        endogenous_effect = endogeneity * cfg.confound_scale * unobserved
        
        # 랜덤 노이즈 (관찰 가능한 마진 변동 — 외생적)
        noise = self.rng.normal(0, cfg.price_noise_std, size=n)
        
        # 최종 마진율
        treatment = base_margin + covariate_effect + endogenous_effect + noise
        
        # 범위 제한
        treatment = np.clip(treatment, cfg.margin_min, cfg.margin_max)
        
        return treatment
    
    def _generate_outcome(
        self,
        base_demands: np.ndarray,
        alphas: np.ndarray,
        treatments: np.ndarray,
        unobserved: np.ndarray,
        endogeneity: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Outcome (수익) 생성
        
        Revenue = demand × margin
        demand = base_demand × exp(-α × margin) × exp(confounding + noise)
        """
        return self._compute_revenue(
            base_demands, alphas, treatments, unobserved, endogeneity
        )
    
    def _compute_revenue(
        self,
        base_demands: np.ndarray,
        alphas: np.ndarray,
        treatments: np.ndarray,
        unobserved: np.ndarray,
        endogeneity: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """수익 계산 (공통 로직)
        
        지수감소 수요 모델:
            demand = base_demand × exp(-α × margin)
                     × exp(endogeneity × confound_scale × unobserved)
                     × exp(noise)
        
        Revenue = demand × margin
        """
        cfg = self.config
        
        # 가격 효과: 지수감소 수요
        price_effect = np.exp(-alphas * treatments)
        
        # 미관측 수요 효과 (내생성)
        confound_effect = np.exp(endogeneity * cfg.confound_scale * unobserved)
        
        # 랜덤 노이즈
        noise = np.exp(self.rng.normal(0, cfg.outcome_noise_std, size=len(treatments)))
        
        # 수요량
        demands = base_demands * price_effect * confound_effect * noise
        demands = np.maximum(demands, 0)
        
        # 수익 = 수요 × 마진
        revenues = demands * treatments
        
        return revenues, demands
    
    def _compute_oracle(
        self,
        base_demands: np.ndarray,
        alphas: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ground truth 최적 마진과 최적 수익 계산
        
        지수감소 수요에서 수익 최대화:
            Revenue(m) = base_demand × m × exp(-α × m)
            dR/dm = base_demand × exp(-αm) × (1 - αm) = 0
            → m* = 1/α
        """
        cfg = self.config
        
        # 최적 마진: m* = 1/α (범위 내로 클리핑)
        optimal_margins = np.clip(1.0 / alphas, cfg.margin_min, cfg.margin_max)
        
        # 최적 수익 (노이즈/교란 없는 기대값)
        optimal_demands = base_demands * np.exp(-alphas * optimal_margins)
        optimal_revenues = optimal_demands * optimal_margins
        
        return optimal_margins, optimal_revenues
    
    def compute_counterfactual_revenue(
        self,
        data: pd.DataFrame,
        policy_margins: np.ndarray,
    ) -> np.ndarray:
        """특정 정책의 반사실 수익 계산 (평가용)
        
        노이즈/교란 없는 기대값으로 계산 (공정한 비교를 위해)
        """
        base_demands = data['base_demand'].values
        alphas = data['true_alpha'].values
        
        demands = base_demands * np.exp(-alphas * policy_margins)
        revenues = demands * policy_margins
        
        return revenues
    
    def get_feature_columns(self) -> list:
        """모델 학습에 사용할 feature 컬럼 목록"""
        return [
            'cat_encoded', 'state_encoded', 'store_encoded',
            'base_sales', 'sales_volatility', 'zero_sales_ratio',
            'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
            'is_event', 'snap',
        ]


# =============================================================================
# Part 3: 편의 함수
# =============================================================================

def create_dataset(
    n_samples: int = 10000,
    endogeneity: float = 0.3,
    heterogeneity: float = 0.5,
    m5_data_dir: Optional[str] = None,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, SemiSyntheticDGP]:
    """한 줄로 데이터셋 생성
    
    Args:
        n_samples: 샘플 수
        endogeneity: 내생성 강도 (0.0 = 없음, 1.0 = 강함)
        heterogeneity: 효과 이질성 (0.0 = 균일, 1.0 = 매우 이질적)
        m5_data_dir: M5 데이터 경로 (None이면 synthetic)
        random_seed: 랜덤 시드
        
    Returns:
        (data, dgp): 생성된 데이터와 DGP 객체
    """
    # Config 설정
    m5_config = M5Config(data_dir=m5_data_dir or "./m5_data")
    dgp_config = DGPConfig(
        n_samples=n_samples,
        endogeneity_strength=endogeneity,
        heterogeneity_strength=heterogeneity,
        random_seed=random_seed,
    )
    
    # Covariates 로드
    covariates = load_m5_covariates(m5_config)
    
    # DGP 생성
    dgp = SemiSyntheticDGP(dgp_config)
    data = dgp.generate(
        covariates,
        n_samples=n_samples,
        endogeneity_strength=endogeneity,
        heterogeneity_strength=heterogeneity,
    )
    
    return data, dgp


# =============================================================================
# Part 4: 진단 & 시각화
# =============================================================================

def diagnose_dataset(data: pd.DataFrame) -> Dict:
    """생성된 데이터셋의 기본 진단
    
    Returns:
        dict with diagnostic results
    """
    diag = {}
    
    # 1. Treatment 분포
    diag['treatment'] = {
        'mean': float(data['treatment'].mean()),
        'std': float(data['treatment'].std()),
        'min': float(data['treatment'].min()),
        'max': float(data['treatment'].max()),
    }
    
    # 2. Outcome 분포
    diag['outcome'] = {
        'mean': float(data['outcome'].mean()),
        'std': float(data['outcome'].std()),
        'min': float(data['outcome'].min()),
        'max': float(data['outcome'].max()),
    }
    
    # 3. 내생성 진단: treatment-outcome 상관
    diag['endogeneity'] = {
        'treatment_outcome_corr': float(data['treatment'].corr(data['outcome'])),
        'treatment_confounder_corr': float(data['treatment'].corr(data['unobserved_confounder'])),
        'outcome_confounder_corr': float(data['outcome'].corr(data['unobserved_confounder'])),
    }
    
    # 4. 가격민감도(α) 분포
    diag['alpha'] = {
        'mean': float(data['true_alpha'].mean()),
        'std': float(data['true_alpha'].std()),
    }
    
    # 카테고리별 α와 최적마진
    for cat in data['cat_id'].unique():
        cat_data = data[data['cat_id'] == cat]
        diag['alpha'][f'{cat}_alpha_mean'] = float(cat_data['true_alpha'].mean())
        diag['alpha'][f'{cat}_optimal_margin_mean'] = float(cat_data['true_optimal_margin'].mean())
    
    # 5. Oracle 대비 baseline 성능
    diag['policy_comparison'] = {
        'baseline_mean_revenue': float(data['baseline_revenue'].mean()),
        'oracle_mean_revenue': float(data['true_optimal_revenue'].mean()),
        'improvement_potential': float(
            (data['true_optimal_revenue'].mean() - data['baseline_revenue'].mean())
            / data['baseline_revenue'].mean() * 100
        ),
    }
    
    return diag


def print_diagnosis(diag: Dict):
    """진단 결과를 보기 좋게 출력"""
    print("\n" + "=" * 60)
    print("  데이터셋 진단 결과")
    print("=" * 60)
    
    print(f"\n📊 Treatment (마진율)")
    t = diag['treatment']
    print(f"   평균: {t['mean']:.3f}, 표준편차: {t['std']:.3f}")
    print(f"   범위: [{t['min']:.3f}, {t['max']:.3f}]")
    
    print(f"\n💰 Outcome (수익)")
    o = diag['outcome']
    print(f"   평균: {o['mean']:.3f}, 표준편차: {o['std']:.3f}")
    
    print(f"\n🔗 내생성 진단")
    e = diag['endogeneity']
    print(f"   Treatment↔Outcome 상관: {e['treatment_outcome_corr']:.3f}")
    print(f"   Treatment↔Confounder 상관: {e['treatment_confounder_corr']:.3f}")
    print(f"   Outcome↔Confounder 상관: {e['outcome_confounder_corr']:.3f}")
    
    # 내생성 해석
    tc = abs(e['treatment_confounder_corr'])
    if tc < 0.05:
        print("   → 내생성 거의 없음 ✓")
    elif tc < 0.15:
        print("   → 약한 내생성")
    elif tc < 0.30:
        print("   → 중간 내생성 ⚠")
    else:
        print("   → 강한 내생성 ⚠⚠")
    
    print(f"\n📈 가격민감도(α) 분포")
    el = diag['alpha']
    print(f"   전체 평균 α: {el['mean']:.2f} (표준편차: {el['std']:.2f})")
    for key, val in el.items():
        if '_alpha_mean' in key:
            cat_name = key.replace('_alpha_mean', '')
            opt_key = f'{cat_name}_optimal_margin_mean'
            opt_val = el.get(opt_key, 0)
            print(f"   {cat_name}: α={val:.2f}, 최적마진={opt_val:.1%}")
    
    print(f"\n🎯 정책 비교")
    p = diag['policy_comparison']
    print(f"   Baseline(15%) 평균 수익: {p['baseline_mean_revenue']:.3f}")
    print(f"   Oracle(최적) 평균 수익: {p['oracle_mean_revenue']:.3f}")
    print(f"   개선 잠재력: {p['improvement_potential']:.1f}%")
    
    print("\n" + "=" * 60)


# =============================================================================
# 실행 예시
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Stage 0: Semi-Synthetic DGP 테스트")
    print("=" * 60)
    
    # 1. 기본 데이터셋 생성 (synthetic covariates — M5 없어도 동작)
    print("\n[1] 기본 데이터셋 생성 (n=5000, endogeneity=0.3)")
    data, dgp = create_dataset(
        n_samples=5000,
        endogeneity=0.3,
        heterogeneity=0.5,
    )
    
    print(f"   shape: {data.shape}")
    print(f"   columns: {list(data.columns)}")
    
    # 2. 진단
    print("\n[2] 데이터셋 진단")
    diag = diagnose_dataset(data)
    print_diagnosis(diag)
    
    # 3. 다양한 내생성 강도에서 비교
    print("\n[3] 내생성 강도별 비교")
    print(f"{'Endogeneity':>12} | {'T↔U Corr':>10} | {'T↔Y Corr':>10} | {'Improvement%':>13}")
    print("-" * 55)
    
    for endo in [0.0, 0.1, 0.3, 0.5, 0.8]:
        test_data, _ = create_dataset(
            n_samples=3000,
            endogeneity=endo,
            heterogeneity=0.5,
        )
        test_diag = diagnose_dataset(test_data)
        e = test_diag['endogeneity']
        p = test_diag['policy_comparison']
        print(f"{endo:>12.1f} | {e['treatment_confounder_corr']:>10.3f} | "
              f"{e['treatment_outcome_corr']:>10.3f} | {p['improvement_potential']:>12.1f}%")
    
    print("\n✓ Stage 0 테스트 완료!")