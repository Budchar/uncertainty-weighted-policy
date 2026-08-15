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
   - B회 반복 추정으로 개체별 최적 처리의 표준편차 σ_B(x)를 산출하며,
     Stage 2의 통계적 불안정성 u_stat 입력이 됨

파이프라인상의 위치:
- 안전 레이어(Stage 3)와 독립적인 표준 추정 단계에 해당한다.
- 논문 실험은 5-fold cross-fitting + XGBoost nuisance model 기반 DML을 사용한다.
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
    """Cross-fitting 기반 dose-response 추정 (비직교)

    실제 동작 — 이름의 "DML"보다 좁은 범위임에 주의:
        1. K-fold cross-fitting으로 nuisance 잔차를 계산한다.
             T_res = T − Ê[T|X],  Y_res = Y − Ê[Y|X]
        2. 그러나 최종 outcome 모델은 원본 (X, T) → Y 로 학습한다.
           위 잔차는 get_residual_diagnostics()의 진단값으로만 쓰이고
           predict() / predict_individual()에는 관여하지 않는다.

        따라서 μ̂(x,t)는 직교화(잔차-on-잔차 회귀)가 적용되지 않은 outcome
        regression(S-learner)이다. Chernozhukov et al.(2018)의 Neyman
        직교 모먼트 조건을 만족하는 추정량이 아니다.

    직교화를 실제로 적용하는 버전은 OrthogonalDMLEstimator를 참조.
    (--estimator odml)

    이 클래스를 수정하지 말 것:
        results/summary_20260416_200920.csv의 360런은 이 구현으로 산출된
        논문 원본 수치다. 동작을 바꾸면 논문의 모든 표가 재현되지 않는다.
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
            'orthogonalized': False,
        }


class OrthogonalDMLEstimator(DoseResponseEstimator):
    """직교화를 실제로 적용하는 DML dose-response 추정기

    DMLDoseResponseEstimator와의 차이:
        기존 DML 추정기는 cross-fitting으로 잔차를 계산하지만 최종 outcome
        모델을 원본 (X, T) → Y 로 학습하므로, 직교화가 추정에 반영되지 않는다.
        이 추정기는 처리 변수를 **잔차화된 기저(basis)로만** 최종 모델에
        투입하여 X를 통한 교란 경로를 실제로 부분화(partial out)한다.

    모형:
        Y = Σ_k θ_k(X)·φ_k(T) + g(X) + ε
        여기서 φ = (T, T², …, T^d) 는 처리 기저이다. 연속 처리에서 단일
        선형항만 쓰면 dose-response가 단조가 되어 내부 최적점이 사라지므로,
        역U자 곡선을 표현할 수 있도록 d ≥ 2 를 기본값으로 둔다.

    추정 절차 (Chernozhukov et al. 2018의 cross-fitting + R-learner 구성):
        1. K-fold cross-fitting으로 nuisance 적합
             ĝ(X)   = E[Y | X]
             ĥ_k(X) = E[φ_k(T) | X]
        2. 잔차화
             Ỹ    = Y − ĝ(X)
             φ̃_k  = φ_k(T) − ĥ_k(X)
        3. 최종 모델을 잔차 위에서 학습
             f̂ :  (X, φ̃_1, …, φ̃_d)  →  Ỹ
           처리 변수는 φ̃ 를 통해서만 들어가므로, X로 설명되는 처리 변동은
           최종 모델이 볼 수 없다.
        4. 예측
             μ̂(x, t) = ĝ(x) + f̂(x, φ_1(t) − ĥ_1(x), …, φ_d(t) − ĥ_d(x))

        잔차 계산에는 cross-fitting 값을 쓰고(과적합 방지), 임의의 t에서
        예측하기 위한 ĝ·ĥ_k 는 전체 표본으로 다시 적합한다. 이는 DML의
        표준 관행이다.

    최종 모델이 X를 함께 받으므로 θ_k(X)가 개체별로 달라질 수 있고,
    따라서 argmax_t μ̂(x,t)의 이질성이 보존된다.
    """

    def __init__(self, n_folds=5, n_estimators=100, max_depth=4, basis_degree=2):
        self.n_folds = n_folds
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.basis_degree = basis_degree
        self.final_model = None
        self.g_model = None
        self.h_models = []
        self.scaler_X = StandardScaler()
        self._Y_residuals = None
        self._T_residuals = None
        self._Phi_residuals = None
        self._is_fitted = False

    def _basis(self, T: np.ndarray) -> np.ndarray:
        """처리 기저 φ(T) = (T, T², …, T^d) — (n, d)"""
        return np.column_stack([T ** k for k in range(1, self.basis_degree + 1)])

    def _make(self, scale=1):
        return make_regressor(
            n_estimators=self.n_estimators * scale,
            max_depth=self.max_depth + (scale - 1),
            random_state=42,
        )

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> 'OrthogonalDMLEstimator':
        n = len(Y)
        d = self.basis_degree
        X_scaled = self.scaler_X.fit_transform(X)
        Phi = self._basis(T)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        Y_res = np.zeros(n)
        Phi_res = np.zeros((n, d))
        T_res = np.zeros(n)   # 진단용 (φ_1 = T 의 잔차와 동일)

        # --- 1~2단계: cross-fitting nuisance 적합 후 잔차화 ---
        for train_idx, test_idx in kf.split(X_scaled):
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]

            y_model = self._make()
            y_model.fit(X_tr, Y[train_idx])
            Y_res[test_idx] = Y[test_idx] - y_model.predict(X_te)

            for k in range(d):
                h_model = self._make()
                h_model.fit(X_tr, Phi[train_idx, k])
                Phi_res[test_idx, k] = Phi[test_idx, k] - h_model.predict(X_te)

        T_res[:] = Phi_res[:, 0]

        self._Y_residuals = Y_res
        self._T_residuals = T_res
        self._Phi_residuals = Phi_res

        # --- 예측용 nuisance: 전체 표본으로 재적합 ---
        self.g_model = self._make()
        self.g_model.fit(X_scaled, Y)

        self.h_models = []
        for k in range(d):
            h_model = self._make()
            h_model.fit(X_scaled, Phi[:, k])
            self.h_models.append(h_model)

        # --- 3단계: 잔차 위에서 최종 모델 학습 ---
        # 처리 변수는 잔차화된 기저 Phi_res 로만 진입한다.
        Z = np.column_stack([X_scaled, Phi_res])
        self.final_model = self._make(scale=2)
        self.final_model.fit(Z, Y_res)

        self._is_fitted = True
        return self

    def _residualized_basis_at(self, X_scaled: np.ndarray, t: float) -> np.ndarray:
        """주어진 t에서의 잔차화 기저 φ_k(t) − ĥ_k(x) — (n, d)"""
        n = len(X_scaled)
        phi_t = self._basis(np.full(n, t))
        h_pred = np.column_stack([hm.predict(X_scaled) for hm in self.h_models])
        return phi_t - h_pred

    def predict(self, X: np.ndarray, t_values: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_X.transform(X)
        n, m = len(X), len(t_values)

        g = self.g_model.predict(X_scaled)
        # ĥ_k(x)는 t에 의존하지 않으므로 한 번만 계산한다.
        h_pred = np.column_stack([hm.predict(X_scaled) for hm in self.h_models])

        predictions = np.zeros((n, m))
        for j, t in enumerate(t_values):
            phi_t = self._basis(np.full(n, t))
            Z = np.column_stack([X_scaled, phi_t - h_pred])
            predictions[:, j] = g + self.final_model.predict(Z)

        return predictions

    def predict_individual(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_X.transform(X)
        g = self.g_model.predict(X_scaled)
        phi = self._basis(T)
        h_pred = np.column_stack([hm.predict(X_scaled) for hm in self.h_models])
        Z = np.column_stack([X_scaled, phi - h_pred])
        return g + self.final_model.predict(Z)

    def get_residual_diagnostics(self) -> Dict:
        if self._T_residuals is None:
            return {}
        return {
            'T_residual_std': float(np.std(self._T_residuals)),
            'Y_residual_std': float(np.std(self._Y_residuals)),
            'T_Y_residual_corr': float(np.corrcoef(
                self._T_residuals, self._Y_residuals
            )[0, 1]),
            'orthogonalized': True,
            'basis_degree': int(self.basis_degree),
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
    elif estimator_type == "odml":
        # 직교화를 실제로 적용하는 DML. nuisance는 dml과 동일 조건(XGBoost)으로
        # 맞춰, 차이가 오직 "잔차화 적용 여부"에서만 오도록 통제한다.
        est_class = OrthogonalDMLEstimator
        est_kwargs = {'n_folds': 5, 'n_estimators': 100, 'max_depth': 4,
                      'basis_degree': 2}
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