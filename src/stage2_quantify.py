"""
Stage 2: Uncertainty Quantification
=====================================

역할: Stage 1에서 추정한 dose-response curve의 불확실성을
      개인 단위로 정량화한다.

방법:
1. Conformal Prediction (CP)
   - 분포 가정 없이 finite-sample 유효한 예측 구간 생성
   - Split conformal: calibration set의 잔차로 구간 폭 결정

2. 불확실성 합성 (Uncertainty Synthesis) — 두 지표 + 상호작용
   - u_stat: "추정이 흔들리는가?" (bootstrap std / treatment range)
   - u_pos:  "해본 적 없는 곳인가?" (GPS 기반 외삽 위험)
   - u_stat × u_pos: "흔들리면서 동시에 관찰도 부족한가?" (상호작용)
   
   합성: u(x) = (u_stat + u_pos + u_stat·u_pos) / 3
   정규화: 절대적 정규화 (물리적 의미 기반, 시나리오 간 비교 가능)

설계 근거:
   두 지표는 서로 다른 축의 epistemic uncertainty를 측정한다.
   - u_stat: 추정 분산(variance) — 데이터가 많으면 줄어듦
   - u_pos: 외삽 위험(extrapolation) — 관찰 밀도가 높으면 줄어듦
   상호작용항은 "추정도 불안정하고 관찰도 부족한" 영역을 
   개별 지표의 합보다 더 강하게 보수적으로 처리한다.
   
   기존 u_sens(곡률/민감도)는 epistemic uncertainty가 아니라 
   의사결정의 결과 민감도(loss sensitivity)로, 추정이 맞았을 때와
   틀렸을 때 반대 방향으로 작동하여 보수성 가중치로 부적합.
   (Ablation 실험에서 시나리오별 부호 반전 확인됨)

핵심 출력:
   u(x_i) → Stage 3에서 π_safe(x) = (1-λu)π* + λu·π₀ 에 사용
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field

from sklearn.model_selection import train_test_split

import sys
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.stage1_estimate import (
    BootstrapDoseResponse,
    DoseResponseEstimator,
    GPSEstimator,
)


# =============================================================================
# Part 1: Conformal Prediction for Dose-Response
# =============================================================================

class ConformalDoseResponse:
    """Split Conformal Prediction for dose-response curves
    
    원리:
    1. 데이터를 train / calibration으로 분할
    2. train에서 dose-response 모델 학습
    3. calibration에서 잔차(nonconformity score) 수집
    4. 새로운 (x, t)에 대해 잔차의 분위수로 예측 구간 생성
    
    장점: 분포 가정 없이 coverage guarantee
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: 목표 miscoverage rate (0.1 → 90% 구간)
        """
        self.alpha = alpha
        self.calibration_scores = None  # nonconformity scores
        self.quantile = None  # 구간 폭 결정 분위수
        self._is_fitted = False
    
    def calibrate(
        self,
        estimator: DoseResponseEstimator,
        X_cal: np.ndarray,
        T_cal: np.ndarray,
        Y_cal: np.ndarray,
    ) -> 'ConformalDoseResponse':
        """Calibration: 잔차 분포에서 예측 구간 폭 결정
        
        Args:
            estimator: 학습된 dose-response 추정기
            X_cal: calibration covariates
            T_cal: calibration treatments
            Y_cal: calibration outcomes
        """
        # Nonconformity scores = |Y - Ŷ|
        Y_pred = estimator.predict_individual(X_cal, T_cal)
        self.calibration_scores = np.abs(Y_cal - Y_pred)
        
        # (1-α)(1+1/n) 분위수 — finite sample correction
        n_cal = len(Y_cal)
        adjusted_quantile = min(
            np.ceil((1 - self.alpha) * (n_cal + 1)) / n_cal, 1.0
        )
        self.quantile = np.quantile(self.calibration_scores, adjusted_quantile)
        
        self._is_fitted = True
        return self
    
    def predict_interval(
        self,
        estimator: DoseResponseEstimator,
        X: np.ndarray,
        T: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """예측 구간 생성
        
        Returns:
            point_est: (n,) 점추정
            lower: (n,) 하한
            upper: (n,) 상한
        """
        point_est = estimator.predict_individual(X, T)
        lower = point_est - self.quantile
        upper = point_est + self.quantile
        return point_est, lower, upper
    
    def get_interval_width(self) -> float:
        """전체 구간 폭 (= 2 × quantile)"""
        return 2 * self.quantile


class AdaptiveConformalDoseResponse:
    """Locally Adaptive Conformal Prediction
    
    기본 CP는 모든 개인에게 동일한 구간 폭을 적용하지만,
    이 버전은 개인별로 다른 폭을 부여한다.
    
    방법: 잔차를 정규화하여 개인별 난이도를 반영
    - 난이도 추정: σ(x) = E[|Y - Ŷ| | X=x]
    - 정규화 잔차: |Y - Ŷ| / σ(x)
    - 개인별 구간: Ŷ ± q × σ(x)
    
    → 예측이 어려운 영역은 넓은 구간, 쉬운 영역은 좁은 구간
    """
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.difficulty_model = None
        self.normalized_quantile = None
        self._is_fitted = False
    
    def calibrate(
        self,
        estimator: DoseResponseEstimator,
        X_cal: np.ndarray,
        T_cal: np.ndarray,
        Y_cal: np.ndarray,
        X_train: np.ndarray,
        T_train: np.ndarray,
        Y_train: np.ndarray,
    ) -> 'AdaptiveConformalDoseResponse':
        """Calibration with difficulty estimation
        
        Args:
            estimator: 학습된 dose-response 추정기
            X/T/Y_cal: calibration 데이터
            X/T/Y_train: 학습 데이터 (difficulty model 학습용)
        """
        from sklearn.ensemble import GradientBoostingRegressor
        
        # Step 1: 학습 데이터에서 잔차의 절대값 모델 학습
        Y_pred_train = estimator.predict_individual(X_train, T_train)
        abs_residuals_train = np.abs(Y_train - Y_pred_train)
        
        # σ(x, t) 모델: 잔차 크기를 예측
        XT_train = np.column_stack([X_train, T_train])
        self.difficulty_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=42
        )
        self.difficulty_model.fit(XT_train, abs_residuals_train)
        
        # Step 2: Calibration 데이터에서 정규화 잔차 계산
        Y_pred_cal = estimator.predict_individual(X_cal, T_cal)
        abs_residuals_cal = np.abs(Y_cal - Y_pred_cal)
        
        XT_cal = np.column_stack([X_cal, T_cal])
        sigma_cal = self.difficulty_model.predict(XT_cal)
        sigma_cal = np.maximum(sigma_cal, 1e-6)  # 0 방지
        
        normalized_scores = abs_residuals_cal / sigma_cal
        
        # Step 3: 정규화 잔차의 분위수
        n_cal = len(Y_cal)
        adjusted_quantile = min(
            np.ceil((1 - self.alpha) * (n_cal + 1)) / n_cal, 1.0
        )
        self.normalized_quantile = np.quantile(normalized_scores, adjusted_quantile)
        
        self._is_fitted = True
        return self
    
    def predict_interval(
        self,
        estimator: DoseResponseEstimator,
        X: np.ndarray,
        T: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """개인별 적응적 예측 구간"""
        point_est = estimator.predict_individual(X, T)
        
        XT = np.column_stack([X, T])
        sigma = self.difficulty_model.predict(XT)
        sigma = np.maximum(sigma, 1e-6)
        
        margin = self.normalized_quantile * sigma
        lower = point_est - margin
        upper = point_est + margin
        
        return point_est, lower, upper
    
    def predict_sigma(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """개인별 예측 난이도(σ) 반환"""
        XT = np.column_stack([X, T])
        sigma = self.difficulty_model.predict(XT)
        return np.maximum(sigma, 1e-6)


# =============================================================================
# Part 2: Uncertainty Synthesis
# =============================================================================

class UncertaintySynthesizer:
    """두 축의 epistemic uncertainty + 상호작용으로 u(x) ∈ [0, 1] 생성
    
    원천 1 — 통계적 불확실성 (u_stat):
        "추정이 흔들리는가?"
        Bootstrap 최적 treatment의 표준편차 / treatment 범위
        → 절대적 정규화: 범위의 몇 %만큼 흔들리는가
    
    원천 2 — Positivity 불확실성 (u_pos):
        "해본 적 없는 곳인가?"
        GPS가 threshold 이하 → 관찰 부족 → 외삽 위험
        → 절대적 정규화: GPS / median(GPS) 기반
    
    상호작용 — (u_stat × u_pos):
        "추정이 흔들리면서 동시에 관찰도 부족한가?"
        개별 지표만으로는 포착 못하는 복합 위험.
        u_stat만 높거나 u_pos만 높은 건 한 축의 문제지만,
        둘 다 높으면 추정의 근거 자체가 취약 → 더 강한 보수성 필요.
    
    합성: u(x) = (u_stat + u_pos + u_stat·u_pos) / 3
    """
    
    def __init__(self):
        pass
    
    def compute_statistical_uncertainty(
        self,
        bootstrap_est: BootstrapDoseResponse,
        X: np.ndarray,
        t_grid: np.ndarray,
    ) -> np.ndarray:
        """Bootstrap 기반 통계적 불확실성
        
        절대적 정규화: optimal_t_std / treatment_range
        직관: treatment 범위(0.30)의 몇 %만큼 흔들리는가
        - std=0.03 → u=0.1 (안정적)
        - std=0.15 → u=0.5 (상당히 흔들림)
        """
        result = bootstrap_est.find_optimal_with_uncertainty(X, t_grid)
        optimal_t_std = result['optimal_t_std']
        
        treatment_range = t_grid[-1] - t_grid[0]  # 0.30
        u_stat = optimal_t_std / treatment_range
        u_stat = np.clip(u_stat, 0, 1)
        
        return u_stat
    
    def compute_positivity_uncertainty(
        self,
        gps_estimator: GPSEstimator,
        X: np.ndarray,
        T_optimal: np.ndarray,
    ) -> np.ndarray:
        """GPS 기반 positivity 불확실성
        
        절대적 정규화: max(1 - gps/threshold, 0)
        - GPS >= threshold → u=0 (충분히 관찰됨)
        - GPS = 0 → u=1 (전혀 관찰 안 됨)
        threshold = median(GPS): 관찰 밀도의 중위수
        """
        gps_at_optimal = gps_estimator.compute_gps(X, T_optimal)
        
        threshold = np.median(gps_at_optimal)
        threshold = max(threshold, 1e-10)  # 0 방지
        
        u_pos = np.maximum(1.0 - gps_at_optimal / threshold, 0.0)
        u_pos = np.clip(u_pos, 0, 1)
        
        return u_pos
    
    def synthesize(
        self,
        u_statistical: np.ndarray,
        u_positivity: np.ndarray,
    ) -> np.ndarray:
        """두 epistemic uncertainty + 상호작용항으로 합성 → u(x) ∈ [0, 1]
        
        u(x) = (u_stat + u_pos + u_stat·u_pos) / 3
        
        상호작용항의 역할:
        - u_stat=0.8, u_pos=0.8 → interaction=0.64, u=0.75 (강한 보수성)
        - u_stat=0.8, u_pos=0.0 → interaction=0.00, u=0.27 (한쪽만 높음)
        - u_stat=0.0, u_pos=0.8 → interaction=0.00, u=0.27 (한쪽만 높음)
        
        두 원천이 동시에 높을 때만 강하게 보수적으로 작동한다.
        """
        interaction = u_statistical * u_positivity
        u = (u_statistical + u_positivity + interaction) / 3.0
        u = np.clip(u, 0, 1)
        return u


# =============================================================================
# Part 3: Stage 2 Runner
# =============================================================================

def run_stage2(
    data: pd.DataFrame,
    feature_cols: List[str],
    bootstrap_est: BootstrapDoseResponse,
    t_grid: np.ndarray,
    alpha: float = 0.1,
    synthesis_method: str = "interaction",
) -> Dict:
    """Stage 2 전체 실행
    
    Args:
        data: Stage 0 데이터
        feature_cols: feature 컬럼들
        bootstrap_est: Stage 1에서 학습된 bootstrap 추정기
        t_grid: treatment grid
        alpha: conformal prediction miscoverage rate
        synthesis_method: 호환성 유지 (기본: interaction)
    
    Returns:
        dict with uncertainty scores and diagnostics
    """
    available_cols = [c for c in feature_cols if c in data.columns]
    X = data[available_cols].values
    T = data['treatment'].values
    Y = data['outcome'].values
    n = len(Y)
    
    print(f"\n[Stage 2] 불확실성 정량화 시작")
    
    # --- Step 1: Conformal Prediction ---
    print(f"  [1/3] Conformal Prediction (α={alpha})...")
    
    # Train/Calibration 분할
    indices = np.arange(n)
    idx_train, idx_cal = train_test_split(indices, test_size=0.3, random_state=42)
    
    X_train, X_cal = X[idx_train], X[idx_cal]
    T_train, T_cal = T[idx_train], T[idx_cal]
    Y_train, Y_cal = Y[idx_train], Y[idx_cal]
    
    # Adaptive Conformal
    adaptive_cp = AdaptiveConformalDoseResponse(alpha=alpha)
    adaptive_cp.calibrate(
        bootstrap_est.main_estimator,
        X_cal, T_cal, Y_cal,
        X_train, T_train, Y_train,
    )
    
    # 전체 데이터에 대한 구간
    point_est, lower, upper = adaptive_cp.predict_interval(
        bootstrap_est.main_estimator, X, T
    )
    coverage = np.mean((Y >= lower) & (Y <= upper))
    mean_width = np.mean(upper - lower)
    
    print(f"    Coverage: {coverage:.1%} (목표: {1-alpha:.0%})")
    print(f"    평균 구간 폭: {mean_width:.4f}")
    
    # --- Step 2: 불확실성 원천별 계산 ---
    print(f"  [2/3] 통계적 불확실성 (Bootstrap)...")
    synthesizer = UncertaintySynthesizer()
    u_stat = synthesizer.compute_statistical_uncertainty(bootstrap_est, X, t_grid)
    
    print(f"  [3/3] Positivity 불확실성 (GPS)...")
    # GPS estimator 접근
    main_est = bootstrap_est.main_estimator
    if hasattr(main_est, 'gps_estimator'):
        gps_est = main_est.gps_estimator
    else:
        # DML의 경우 별도 GPS 학습
        gps_est = GPSEstimator()
        gps_est.fit(X, T)
    
    # 추정 최적 treatment
    opt_result = bootstrap_est.find_optimal_with_uncertainty(X, t_grid)
    optimal_t = opt_result['optimal_t']
    
    u_pos = synthesizer.compute_positivity_uncertainty(gps_est, X, optimal_t)
    
    # --- Step 3: 합성 (두 원천 + 상호작용) ---
    u_combined = synthesizer.synthesize(u_stat, u_pos)
    interaction = u_stat * u_pos
    
    # --- 결과 정리 ---
    results = {
        # 불확실성 점수
        'u_combined': u_combined,
        'u_statistical': u_stat,
        'u_positivity': u_pos,
        'u_interaction': interaction,
        
        # 추정 최적 treatment
        'optimal_t': optimal_t,
        'optimal_t_std': opt_result['optimal_t_std'],
        
        # Conformal 진단
        'conformal_coverage': float(coverage),
        'conformal_mean_width': float(mean_width),
        'adaptive_cp': adaptive_cp,
        
        # GPS 진단
        'gps_estimator': gps_est,
        
        # 요약 통계
        'diagnostics': {
            'u_combined_mean': float(u_combined.mean()),
            'u_combined_std': float(u_combined.std()),
            'u_stat_mean': float(u_stat.mean()),
            'u_pos_mean': float(u_pos.mean()),
            'u_interaction_mean': float(interaction.mean()),
            'high_uncertainty_ratio': float(np.mean(u_combined > 0.7)),
            'low_uncertainty_ratio': float(np.mean(u_combined < 0.3)),
        },
    }
    
    return results


def print_stage2_results(results: Dict):
    """Stage 2 결과 출력"""
    diag = results['diagnostics']
    
    print(f"\n{'=' * 60}")
    print(f"  Stage 2 결과: 불확실성 정량화")
    print(f"{'=' * 60}")
    
    print(f"\n🎯 Conformal Prediction")
    print(f"   Coverage: {results['conformal_coverage']:.1%}")
    print(f"   평균 구간 폭: {results['conformal_mean_width']:.4f}")
    
    print(f"\n📊 불확실성 원천별 평균")
    print(f"   통계적 (Bootstrap std):  {diag['u_stat_mean']:.3f}")
    print(f"   Positivity (GPS):        {diag['u_pos_mean']:.3f}")
    print(f"   상호작용 (stat × pos):   {diag['u_interaction_mean']:.3f}")
    
    print(f"\n🔗 종합 불확실성 u(x) = (stat + pos + stat·pos) / 3")
    print(f"   평균: {diag['u_combined_mean']:.3f}")
    print(f"   표준편차: {diag['u_combined_std']:.3f}")
    print(f"   높은 불확실성 (>0.7): {diag['high_uncertainty_ratio']:.1%}")
    print(f"   낮은 불확실성 (<0.3): {diag['low_uncertainty_ratio']:.1%}")
    
    print(f"\n{'=' * 60}")


# =============================================================================
# Part 4: Main
# =============================================================================

if __name__ == "__main__":
    from src.stage0_dgp import create_dataset
    from src.stage1_estimate import run_stage1, print_evaluation
    
    print("=" * 60)
    print("  Stage 2: Uncertainty Quantification 테스트")
    print("=" * 60)
    
    # 1. 데이터 생성
    print("\n[0] 데이터 생성")
    data, dgp = create_dataset(n_samples=3000, endogeneity=0.3, heterogeneity=0.5)
    feature_cols = dgp.get_feature_columns()
    
    # 2. Stage 1: 추정
    bootstrap_dml, eval_dml = run_stage1(
        data, feature_cols, n_bootstrap=20, estimator_type="dml"
    )
    print_evaluation(eval_dml, "DML")
    
    # 3. Stage 2: 불확실성
    t_grid = np.linspace(0.05, 0.35, 31)
    stage2_results = run_stage2(
        data, feature_cols, bootstrap_dml, t_grid,
        alpha=0.1,
    )
    print_stage2_results(stage2_results)
    
    # 4. 불확실성과 실제 추정 오차의 관계
    print("\n[검증] 불확실성이 높은 곳에서 실제로 추정 오차가 큰가?")
    u = stage2_results['u_combined']
    est_optimal = stage2_results['optimal_t']
    true_optimal = data['true_optimal_margin'].values
    abs_error = np.abs(est_optimal - true_optimal)
    
    # 불확실성 구간별 실제 오차
    for label, low, high in [("낮음", 0, 0.3), ("중간", 0.3, 0.7), ("높음", 0.7, 1.01)]:
        mask = (u >= low) & (u < high)
        if mask.sum() > 0:
            print(f"   u={label} ({mask.sum():>4}명): "
                  f"평균 오차={abs_error[mask].mean():.4f}, "
                  f"Safety위반={((data['baseline_revenue'].values > (data['base_demand'].values * np.exp(-data['true_alpha'].values * est_optimal) * est_optimal))[mask]).mean():.1%}")
    
    print("\n✓ Stage 2 테스트 완료!")