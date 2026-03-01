"""
Stage 1: Dose-Response Curve Estimation
========================================

역할: 관찰 데이터에서 마진율(treatment)과 수익(outcome) 간의
      인과적 dose-response 관계를 추정한다.

방법:
1. GPS (Generalized Propensity Score) 추정
   - treatment의 조건부 밀도 f(T|X)를 모델링
   - 정규분포 가정: T|X ~ N(μ(X), σ²(X))

2. Dose-Response Curve 추정
   - GPS를 포함한 outcome 모델: E[Y|T=t, GPS=g(t,x)]
   - 또는 DML 스타일: (X, T)를 함께 넣은 모델

3. Bootstrap Confidence Intervals
   - 개인별 dose-response의 불확실성 정량화
   - Stage 2 (Conformal Prediction)의 입력이 됨

파이프라인상의 위치:
- 표준 인과추정 파이프라인의 추정 단계에 해당
- 대안 추정기로 Regression, S-learner, AIPTW 등이 있음
- 우리는 GPS regression + DML을 사용 (더 표준적)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

from utils.gpu_utils import make_regressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

import sys
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from configs.default_config import DGPConfig


# =============================================================================
# Part 1: GPS (Generalized Propensity Score) Estimation
# =============================================================================

class GPSEstimator:
    """Generalized Propensity Score 추정기
    
    Continuous treatment에 대한 propensity score.
    T|X ~ N(μ(X), σ²) 가정 하에 GPS = f(t|x) = φ((t-μ(x))/σ) / σ
    
    GPS의 역할:
    - 교란 보정: X가 같아도 T 분포가 다른 편향을 보정
    - Positivity 진단: GPS가 극단적으로 작으면 해당 (x,t) 조합의 관찰이 부족
    """
    
    def __init__(self, n_estimators=100, max_depth=4):
        self.treatment_model = make_regressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        self.sigma = None
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, T: np.ndarray) -> 'GPSEstimator':
        """Treatment model 학습: E[T|X] 추정"""
        X_scaled = self.scaler.fit_transform(X)
        self.treatment_model.fit(X_scaled, T)
        
        T_pred = self.treatment_model.predict(X_scaled)
        residuals = T - T_pred
        self.sigma = np.std(residuals)
        
        self._is_fitted = True
        return self
    
    def predict_treatment(self, X: np.ndarray) -> np.ndarray:
        """E[T|X] 예측"""
        X_scaled = self.scaler.transform(X)
        return self.treatment_model.predict(X_scaled)
    
    def compute_gps(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """GPS 값 계산: f(t|x)"""
        T_pred = self.predict_treatment(X)
        gps = norm.pdf(T, loc=T_pred, scale=self.sigma)
        return gps
    
    def compute_residuals(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Treatment 잔차: T - E[T|X]"""
        return T - self.predict_treatment(X)
    
    def diagnose_positivity(
        self, X: np.ndarray, T: np.ndarray, threshold: float = 0.01
    ) -> Dict:
        """Positivity 진단"""
        gps = self.compute_gps(X, T)
        return {
            'gps_mean': float(np.mean(gps)),
            'gps_min': float(np.min(gps)),
            'gps_5pct': float(np.percentile(gps, 5)),
            'low_gps_ratio': float(np.mean(gps < threshold)),
            'treatment_model_r2': float(self.treatment_model.score(
                self.scaler.transform(X), T
            )),
        }


# =============================================================================
# Part 2: Dose-Response Estimators
# =============================================================================

class DoseResponseEstimator:
    """Dose-response curve 추정 — base class"""
    
    def fit(self, X, T, Y):
        raise NotImplementedError
    
    def predict(self, X, t_values):
        """주어진 X에 대해 다양한 treatment level에서의 E[Y|X,T=t] 예측
        
        Returns: (n, m) — 각 개인 × 각 treatment level
        """
        raise NotImplementedError
    
    def predict_individual(self, X, T):
        """각 개인의 관찰된 treatment에서의 E[Y|X,T] 예측"""
        raise NotImplementedError
    
    def find_optimal_treatment(self, X, t_grid):
        """각 개인의 추정 최적 treatment 찾기"""
        predictions = self.predict(X, t_grid)
        best_idx = np.argmax(predictions, axis=1)
        optimal_t = t_grid[best_idx]
        optimal_y = predictions[np.arange(len(X)), best_idx]
        return optimal_t, optimal_y


class GPSRegressionEstimator(DoseResponseEstimator):
    """GPS Regression 기반 dose-response 추정
    
    Hirano & Imbens (2004) 스타일:
    Step 1: GPS = f(T|X) 추정
    Step 2: E[Y|T, GPS] 추정 — outcome을 (T, GPS)로 회귀
    Step 3: dose-response = E_X[E[Y|T=t, GPS=f(t|X)]]
    """
    
    def __init__(self, n_estimators=200, max_depth=5):
        self.gps_estimator = GPSEstimator()
        self.outcome_model = make_regressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        self.scaler_outcome = StandardScaler()
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'GPSRegressionEstimator':
        # Step 1: GPS 추정
        self.gps_estimator.fit(X, T)
        gps = self.gps_estimator.compute_gps(X, T)
        
        # Step 2: Outcome 모델 — features = (T, GPS)
        outcome_features = np.column_stack([T, gps])
        outcome_features_scaled = self.scaler_outcome.fit_transform(outcome_features)
        self.outcome_model.fit(outcome_features_scaled, Y)
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, t_values: np.ndarray) -> np.ndarray:
        n = len(X)
        m = len(t_values)
        predictions = np.zeros((n, m))
        
        for j, t in enumerate(t_values):
            t_array = np.full(n, t)
            gps = self.gps_estimator.compute_gps(X, t_array)
            features = np.column_stack([t_array, gps])
            features_scaled = self.scaler_outcome.transform(features)
            predictions[:, j] = self.outcome_model.predict(features_scaled)
        
        return predictions
    
    def predict_individual(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        gps = self.gps_estimator.compute_gps(X, T)
        features = np.column_stack([T, gps])
        features_scaled = self.scaler_outcome.transform(features)
        return self.outcome_model.predict(features_scaled)


class DMLDoseResponseEstimator(DoseResponseEstimator):
    """Double Machine Learning 기반 dose-response 추정
    
    Chernozhukov et al. (2018) 스타일 — continuous treatment 적용:
    Cross-fitting으로 nuisance 추정 후, (X, T) → Y 모델 학습
    """
    
    def __init__(self, n_folds=5, n_estimators=100, max_depth=4):
        self.n_folds = n_folds
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.effect_model = None
        self.scaler_X = StandardScaler()
        self._Y_residuals = None
        self._T_residuals = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'DMLDoseResponseEstimator':
        n = len(Y)
        X_scaled = self.scaler_X.fit_transform(X)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        Y_residuals = np.zeros(n)
        T_residuals = np.zeros(n)
        
        # Cross-fitting
        for train_idx, test_idx in kf.split(X_scaled):
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
            T_tr, T_te = T[train_idx], T[test_idx]
            Y_tr, Y_te = Y[train_idx], Y[test_idx]
            
            t_model = make_regressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=42
            )
            t_model.fit(X_tr, T_tr)
            T_residuals[test_idx] = T_te - t_model.predict(X_te)
            
            y_model = make_regressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=42
            )
            y_model.fit(X_tr, Y_tr)
            Y_residuals[test_idx] = Y_te - y_model.predict(X_te)
        
        self._Y_residuals = Y_residuals
        self._T_residuals = T_residuals
        
        # Final model: (X, T) → Y
        self.effect_model = make_regressor(
            n_estimators=self.n_estimators * 2, max_depth=self.max_depth + 1,
            random_state=42
        )
        XT = np.column_stack([X_scaled, T])
        self.effect_model.fit(XT, Y)
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, t_values: np.ndarray) -> np.ndarray:
        n = len(X)
        m = len(t_values)
        X_scaled = self.scaler_X.transform(X)
        predictions = np.zeros((n, m))
        
        for j, t in enumerate(t_values):
            XT = np.column_stack([X_scaled, np.full(n, t)])
            predictions[:, j] = self.effect_model.predict(XT)
        
        return predictions
    
    def predict_individual(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_X.transform(X)
        XT = np.column_stack([X_scaled, T])
        return self.effect_model.predict(XT)
    
    def get_residual_diagnostics(self) -> Dict:
        if self._T_residuals is None:
            return {}
        return {
            'T_residual_std': float(np.std(self._T_residuals)),
            'Y_residual_std': float(np.std(self._Y_residuals)),
            'T_Y_residual_corr': float(np.corrcoef(
                self._T_residuals, self._Y_residuals
            )[0, 1]),
        }


# =============================================================================
# Part 3: Bootstrap Confidence Intervals
# =============================================================================

class BootstrapDoseResponse:
    """Bootstrap으로 dose-response curve의 신뢰구간 추정"""
    
    def __init__(
        self,
        estimator_class: type = GPSRegressionEstimator,
        n_bootstrap: int = 50,
        random_seed: int = 42,
        **estimator_kwargs,
    ):
        self.estimator_class = estimator_class
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(random_seed)
        self.estimator_kwargs = estimator_kwargs
        self.bootstrap_estimators: List[DoseResponseEstimator] = []
        self.main_estimator: Optional[DoseResponseEstimator] = None
    
    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'BootstrapDoseResponse':
        n = len(Y)
        
        print(f"  Main estimator 학습 중...")
        self.main_estimator = self.estimator_class(**self.estimator_kwargs)
        self.main_estimator.fit(X, T, Y)
        
        self.bootstrap_estimators = []
        for b in range(self.n_bootstrap):
            if (b + 1) % 10 == 0:
                print(f"  Bootstrap {b+1}/{self.n_bootstrap}...")
            
            idx = self.rng.choice(n, size=n, replace=True)
            est = self.estimator_class(**self.estimator_kwargs)
            est.fit(X[idx], T[idx], Y[idx])
            self.bootstrap_estimators.append(est)
        
        return self
    
    def predict_with_ci(
        self, X: np.ndarray, t_values: np.ndarray, alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """신뢰구간 포함 예측
        
        Returns: (point_est, ci_lower, ci_upper, ci_width) — 각 (n, m)
        """
        point_est = self.main_estimator.predict(X, t_values)
        
        bootstrap_preds = np.zeros((self.n_bootstrap, len(X), len(t_values)))
        for b, est in enumerate(self.bootstrap_estimators):
            bootstrap_preds[b] = est.predict(X, t_values)
        
        ci_lower = np.percentile(bootstrap_preds, 100 * alpha / 2, axis=0)
        ci_upper = np.percentile(bootstrap_preds, 100 * (1 - alpha / 2), axis=0)
        ci_width = ci_upper - ci_lower
        
        return point_est, ci_lower, ci_upper, ci_width
    
    def find_optimal_with_uncertainty(
        self, X: np.ndarray, t_grid: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """최적 treatment + 불확실성 정보"""
        point_est = self.main_estimator.predict(X, t_grid)
        best_idx = np.argmax(point_est, axis=1)
        optimal_t = t_grid[best_idx]
        optimal_y = point_est[np.arange(len(X)), best_idx]
        
        bootstrap_optimal_t = np.zeros((self.n_bootstrap, len(X)))
        for b, est in enumerate(self.bootstrap_estimators):
            preds_b = est.predict(X, t_grid)
            best_idx_b = np.argmax(preds_b, axis=1)
            bootstrap_optimal_t[b] = t_grid[best_idx_b]
        
        optimal_t_std = np.std(bootstrap_optimal_t, axis=0)
        
        _, _, _, ci_width = self.predict_with_ci(X, t_grid)
        uncertainty_at_optimal = ci_width[np.arange(len(X)), best_idx]
        
        return {
            'optimal_t': optimal_t,
            'optimal_y': optimal_y,
            'optimal_t_std': optimal_t_std,
            'uncertainty_at_optimal': uncertainty_at_optimal,
        }


# =============================================================================
# Part 4: Evaluation
# =============================================================================

def evaluate_estimation(
    data: pd.DataFrame,
    estimator: DoseResponseEstimator,
    feature_cols: List[str],
    t_grid: np.ndarray,
) -> Dict:
    """추정 품질 평가 (ground truth 대비)"""
    X = data[feature_cols].values
    
    est_optimal_t, est_optimal_y = estimator.find_optimal_treatment(X, t_grid)
    true_optimal_t = data['true_optimal_margin'].values
    
    optimal_t_error = est_optimal_t - true_optimal_t
    optimal_t_mae = np.mean(np.abs(optimal_t_error))
    optimal_t_bias = np.mean(optimal_t_error)
    
    # 추정 정책의 실제 수익 (counterfactual)
    true_alphas = data['true_alpha'].values
    base_demands = data['base_demand'].values
    
    est_policy_revenue = base_demands * np.exp(-true_alphas * est_optimal_t) * est_optimal_t
    oracle_revenue = data['true_optimal_revenue'].values
    baseline_revenue = base_demands * np.exp(-true_alphas * 0.15) * 0.15
    
    regret = oracle_revenue - est_policy_revenue
    safety_violation = (est_policy_revenue < baseline_revenue)
    
    cat_results = {}
    for cat in data['cat_id'].unique():
        mask = data['cat_id'] == cat
        cat_results[cat] = {
            'optimal_t_mae': float(np.mean(np.abs(optimal_t_error[mask]))),
            'optimal_t_bias': float(np.mean(optimal_t_error[mask])),
            'mean_regret': float(np.mean(regret[mask])),
            'safety_violation_rate': float(np.mean(safety_violation[mask])),
        }
    
    return {
        'optimal_t_mae': float(optimal_t_mae),
        'optimal_t_bias': float(optimal_t_bias),
        'mean_regret': float(np.mean(regret)),
        'safety_violation_rate': float(np.mean(safety_violation)),
        'est_mean_revenue': float(np.mean(est_policy_revenue)),
        'oracle_mean_revenue': float(np.mean(oracle_revenue)),
        'baseline_mean_revenue': float(np.mean(baseline_revenue)),
        'improvement_over_baseline': float(
            (np.mean(est_policy_revenue) / np.mean(baseline_revenue) - 1) * 100
        ),
        'by_category': cat_results,
    }


def print_evaluation(eval_results: Dict, estimator_name: str = ""):
    """평가 결과 출력"""
    print(f"\n{'=' * 60}")
    print(f"  Stage 1 평가{f' ({estimator_name})' if estimator_name else ''}")
    print(f"{'=' * 60}")
    
    print(f"\n📊 최적 Treatment 추정 정확도")
    print(f"   MAE: {eval_results['optimal_t_mae']:.4f} "
          f"({eval_results['optimal_t_mae']*100:.1f}%p)")
    print(f"   Bias: {eval_results['optimal_t_bias']:+.4f}")
    
    print(f"\n💰 수익 비교")
    print(f"   Baseline(15%): {eval_results['baseline_mean_revenue']:.4f}")
    print(f"   추정 정책:     {eval_results['est_mean_revenue']:.4f}")
    print(f"   Oracle:        {eval_results['oracle_mean_revenue']:.4f}")
    print(f"   개선율:        {eval_results['improvement_over_baseline']:+.1f}%")
    
    print(f"\n⚠️  안전성")
    print(f"   Regret (평균): {eval_results['mean_regret']:.6f}")
    print(f"   Safety 위반율: {eval_results['safety_violation_rate']:.1%}")
    
    print(f"\n📈 카테고리별")
    for cat, cr in eval_results['by_category'].items():
        print(f"   {cat}: MAE={cr['optimal_t_mae']:.4f}, "
              f"위반={cr['safety_violation_rate']:.1%}")
    
    print(f"\n{'=' * 60}")


# =============================================================================
# Part 5: Main
# =============================================================================

def run_stage1(
    data: pd.DataFrame,
    feature_cols: List[str],
    n_bootstrap: int = 30,
    estimator_type: str = "gps",
) -> Tuple[BootstrapDoseResponse, Dict]:
    """Stage 1 전체 실행"""
    available_cols = [c for c in feature_cols if c in data.columns]
    X = data[available_cols].values
    T = data['treatment'].values
    Y = data['outcome'].values
    t_grid = np.linspace(0.05, 0.35, 31)
    
    if estimator_type == "gps":
        est_class = GPSRegressionEstimator
        est_kwargs = {'n_estimators': 200, 'max_depth': 5}
    else:
        est_class = DMLDoseResponseEstimator
        est_kwargs = {'n_folds': 5, 'n_estimators': 100, 'max_depth': 4}
    
    print(f"\n[Stage 1] {estimator_type.upper()} 추정 시작 (bootstrap={n_bootstrap})")
    bootstrap_est = BootstrapDoseResponse(
        estimator_class=est_class,
        n_bootstrap=n_bootstrap,
        **est_kwargs,
    )
    bootstrap_est.fit(X, T, Y)
    
    eval_results = evaluate_estimation(
        data, bootstrap_est.main_estimator, available_cols, t_grid
    )
    
    return bootstrap_est, eval_results


if __name__ == "__main__":
    from src.stage0_dgp import create_dataset
    
    print("=" * 60)
    print("  Stage 1: Dose-Response Estimation 테스트")
    print("=" * 60)
    
    # 1. 데이터 생성
    print("\n[0] 데이터 생성")
    data, dgp = create_dataset(
        n_samples=3000,
        endogeneity=0.3,
        heterogeneity=0.5,
    )
    feature_cols = dgp.get_feature_columns()
    
    # 2. GPS Regression
    bootstrap_gps, eval_gps = run_stage1(
        data, feature_cols, n_bootstrap=20, estimator_type="gps"
    )
    print_evaluation(eval_gps, "GPS Regression")
    
    # 3. DML
    bootstrap_dml, eval_dml = run_stage1(
        data, feature_cols, n_bootstrap=20, estimator_type="dml"
    )
    print_evaluation(eval_dml, "DML")
    
    # 4. 내생성 없는 경우
    print("\n[비교] 내생성 없는 데이터")
    data_clean, _ = create_dataset(
        n_samples=3000, endogeneity=0.0, heterogeneity=0.5,
    )
    _, eval_clean = run_stage1(
        data_clean, feature_cols, n_bootstrap=10, estimator_type="gps"
    )
    print_evaluation(eval_clean, "GPS (내생성 없음)")
    
    # 5. 요약
    print(f"\n{'=' * 60}")
    print(f"  요약: 내생성이 추정에 미치는 영향")
    print(f"{'=' * 60}")
    print(f"{'':>20} | {'내생성 0.0':>12} | {'내생성 0.3':>12}")
    print(f"{'-'*20}-+-{'-'*12}-+-{'-'*12}")
    print(f"{'최적T MAE':>20} | {eval_clean['optimal_t_mae']:>12.4f} | {eval_gps['optimal_t_mae']:>12.4f}")
    print(f"{'Safety 위반율':>20} | {eval_clean['safety_violation_rate']:>12.1%} | {eval_gps['safety_violation_rate']:>12.1%}")
    print(f"{'개선율':>20} | {eval_clean['improvement_over_baseline']:>11.1f}% | {eval_gps['improvement_over_baseline']:>11.1f}%")
    
    print("\n✓ Stage 1 테스트 완료!")