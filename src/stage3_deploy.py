"""
Stage 3: Safe Policy Deployment
=================================

역할: Stage 2에서 산출한 불확실성 u(x)를 이용하여
      안전한 정책을 구성하고 배포한다.

핵심 공식:
    π_safe(x) = (1 - w(x)) · π̂*(x) + w(x) · π₀(x),   w(x) = min(λ·u(x), 1)

    - π̂*(x): 추정 최적 정책 (Stage 1)
    - π₀(x): Baseline 정책 (현재 운영 중인 고정 마진)
    - u(x): 종합 불안정성 (Stage 2, 0~1)
    - λ: 전역 보수성 파라미터 (0 이상. 논문 실험은 0.0~5.0을 0.1 간격 sweep)
    - w(x): 보간 가중치. λ·u(x)는 1을 넘을 수 있고 그대로 쓰면 baseline을
            지나쳐 외삽하므로 min(·, 1)로 클리핑한다.

동작 원리:
    - u(x)=0 (안정): π_safe = π̂* (추정 최적 그대로)
    - w(x)=1 (λ·u(x) ≥ 1): π_safe = π₀ (baseline으로 완전 회귀)
    - λ=0: 보수성 없음 (항상 π̂*) → naive_optimal과 동일

비교 대상 (baselines):
    1. naive_optimal: π* 그대로 적용 (보수성 없음)
    2. uniform_conservative: 모든 개인에 동일한 보수성
    3. threshold_based: u(x) > 임계값이면 baseline, 아니면 π*
    4. uncertainty_weighted: 핵심 제안 — 연속적 보간
    5. oracle: ground truth 최적 (성능 상한)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

import sys
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# =============================================================================
# Part 1: Policy Implementations
# =============================================================================

def policy_naive_optimal(
    optimal_t: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Naive: 추정 최적 treatment 그대로 적용"""
    return optimal_t.copy()


def policy_uniform_conservative(
    optimal_t: np.ndarray,
    baseline_t: float = 0.15,
    conservatism: float = 0.5,
    **kwargs,
) -> np.ndarray:
    """Uniform: 모든 개인에 동일한 보수성 적용
    
    π(x) = (1 - conservatism) · π*(x) + conservatism · π₀
    """
    return (1 - conservatism) * optimal_t + conservatism * baseline_t


def policy_threshold_based(
    optimal_t: np.ndarray,
    uncertainty: np.ndarray,
    baseline_t: float = 0.15,
    threshold: float = 0.5,
    **kwargs,
) -> np.ndarray:
    """Threshold: 불확실성 임계값으로 이진 결정
    
    u(x) > threshold → baseline
    u(x) ≤ threshold → π*(x)
    """
    policy = optimal_t.copy()
    policy[uncertainty > threshold] = baseline_t
    return policy


def policy_uncertainty_weighted(
    optimal_t: np.ndarray,
    uncertainty: np.ndarray,
    baseline_t: float = 0.15,
    lambda_param: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """핵심 제안: Uncertainty-Weighted Policy Interpolation

    π_safe(x) = (1 - w(x)) · π̂*(x) + w(x) · π₀,   w(x) = min(λ·u(x), 1)

    - w(x)가 보간 가중치. λ·u(x)를 [0, 1]로 클리핑한 값이다.
      (클리핑이 없으면 λ·u(x) > 1인 개체가 baseline을 지나쳐 외삽된다)
    - u(x)=0: π̂* 그대로
    - λ·u(x) ≥ 1: π₀로 완전 회귀
    """
    weight = lambda_param * uncertainty
    weight = np.clip(weight, 0, 1)
    return (1 - weight) * optimal_t + weight * baseline_t


# =============================================================================
# Part 2: Evaluation
# =============================================================================

def evaluate_policy(
    data: pd.DataFrame,
    policy_treatments: np.ndarray,
    policy_name: str,
    baseline_t: float = 0.15,
) -> Dict:
    """정책 평가 — ground truth 기반 (노이즈 없는 기대값으로 비교)

    Args:
        baseline_t: 위반율·개선율의 기준이 되는 현행 정책 π₀의 마진율.
                    run_stage3의 baseline_t와 반드시 동일해야 한다.
    """
    true_alphas = data['true_alpha'].values
    base_demands = data['base_demand'].values
    oracle_revenue = data['true_optimal_revenue'].values

    # 모든 정책을 동일 기준(노이즈 없는 기대값)으로 비교
    policy_revenue = base_demands * np.exp(-true_alphas * policy_treatments) * policy_treatments
    baseline_revenue = base_demands * np.exp(-true_alphas * baseline_t) * baseline_t
    
    regret = oracle_revenue - policy_revenue
    safety_violation = policy_revenue < baseline_revenue
    revenue_shortfall = np.maximum(baseline_revenue - policy_revenue, 0)
    
    return {
        'policy_name': policy_name,
        'mean_revenue': float(np.mean(policy_revenue)),
        'mean_regret': float(np.mean(regret)),
        'median_regret': float(np.median(regret)),
        'safety_violation_rate': float(np.mean(safety_violation)),
        'mean_revenue_shortfall': float(np.mean(revenue_shortfall)),
        'improvement_over_baseline': float(
            (np.mean(policy_revenue) / np.mean(baseline_revenue) - 1) * 100
        ),
    }


# =============================================================================
# Part 3: Stage 3 Runner
# =============================================================================

def run_stage3(
    data: pd.DataFrame,
    stage2_results: Dict,
    baseline_t: float = 0.15,
    lambda_values: List[float] = None,
    uniform_values: List[float] = None,
    threshold_values: List[float] = None
) -> Dict:
    """Stage 3 전체 실행 — 5개 정책 비교
    
    Args:
        data: Stage 0 데이터
        stage2_results: Stage 2 출력 (optimal_t, u_combined 등)
        baseline_t: baseline 마진율
        lambda_values: uncertainty_weighted의 λ 후보
        
    Returns:
        all_results: 정책별 평가 결과
    """
    lambda_values = lambda_values or [0.0, 0.3, 0.5, 0.7, 1.0]
    uniform_values = uniform_values or [0.3, 0.5, 0.7]
    threshold_values = threshold_values or [0.3, 0.5, 0.7]

    optimal_t = stage2_results['optimal_t']
    uncertainty = stage2_results['u_combined']
    true_optimal_t = data['true_optimal_margin'].values
    
    print(f"\n[Stage 3] 정책 비교 시작")
    
    all_results = {}
    
    # 1. Oracle
    eval_oracle = evaluate_policy(data, true_optimal_t, "oracle", baseline_t)
    all_results['oracle'] = eval_oracle

    # 2. Baseline (고정 15%)
    eval_baseline = evaluate_policy(
        data, np.full(len(data), baseline_t), "baseline", baseline_t
    )
    all_results['baseline'] = eval_baseline

    # 3. Naive optimal
    eval_naive = evaluate_policy(
        data, policy_naive_optimal(optimal_t), "naive_optimal", baseline_t
    )
    all_results['naive_optimal'] = eval_naive

    # 4. Uniform conservative
    for c in uniform_values:
        t = policy_uniform_conservative(optimal_t, baseline_t, conservatism=c)
        key = f"uniform_c{c}"
        all_results[key] = evaluate_policy(data, t, key, baseline_t)

    # 5. Threshold based
    for th in threshold_values:
        t = policy_threshold_based(optimal_t, uncertainty, baseline_t, threshold=th)
        key = f"threshold_{th}"
        all_results[key] = evaluate_policy(data, t, key, baseline_t)

    # 6. Uncertainty weighted (핵심)
    for lam in lambda_values:
        t = policy_uncertainty_weighted(optimal_t, uncertainty, baseline_t, lambda_param=lam)
        key = f"uw_lambda{lam}"
        all_results[key] = evaluate_policy(data, t, key, baseline_t)
    
    return all_results


def print_stage3_results(all_results: Dict):
    """Stage 3 결과 요약 출력"""
    print(f"\n{'=' * 80}")
    print(f"  Stage 3 결과: 정책 비교")
    print(f"{'=' * 80}")
    
    print(f"\n{'정책':<25} | {'평균수익':>8} | {'개선율':>7} | {'위반율':>6} | {'Regret':>8}")
    print(f"{'-'*25}-+-{'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*8}")
    
    # 정렬: safety violation이 낮은 순
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['safety_violation_rate'])
    
    for name, r in sorted_results:
        print(f"{r['policy_name']:<25} | {r['mean_revenue']:>8.4f} | "
              f"{r['improvement_over_baseline']:>+6.1f}% | "
              f"{r['safety_violation_rate']:>5.1%} | {r['mean_regret']:>8.5f}")
    
    print(f"\n{'=' * 80}")
    
    # 핵심 비교: naive vs best uncertainty_weighted (λ > 0)
    naive = all_results.get('naive_optimal', {})
    best_uw = None
    best_uw_name = None
    
    # 선택 기준: naive 대비 개선율의 70% 이상 유지하면서 위반율이 가장 낮은 것
    # → "수익을 크게 희생하지 않으면서 안전성을 최대한 높이자"
    naive_imp = naive.get('improvement_over_baseline', 0)
    min_acceptable_imp = naive_imp * 0.5  # 최소 naive의 50% 개선 유지
    
    for name, r in all_results.items():
        if not name.startswith('uw_lambda') or name == 'uw_lambda0.0':
            continue
        if r['improvement_over_baseline'] >= min_acceptable_imp:
            if best_uw is None or r['safety_violation_rate'] < best_uw['safety_violation_rate']:
                best_uw = r
                best_uw_name = name
    
    if naive and best_uw:
        print(f"\n🎯 핵심 비교: Naive vs Uncertainty-Weighted ({best_uw_name})")
        print(f"   Naive:  개선={naive['improvement_over_baseline']:+.1f}%, "
              f"위반={naive['safety_violation_rate']:.1%}")
        print(f"   UW:     개선={best_uw['improvement_over_baseline']:+.1f}%, "
              f"위반={best_uw['safety_violation_rate']:.1%}")
        
        viol_reduction = (1 - best_uw['safety_violation_rate'] / max(naive['safety_violation_rate'], 1e-8)) * 100
        print(f"   → 위반율 {viol_reduction:.0f}% 감소")


# =============================================================================
# Part 4: Main
# =============================================================================

if __name__ == "__main__":
    from src.stage0_dgp import create_dataset
    from src.stage1_estimate import run_stage1, print_evaluation
    from src.stage2_quantify import run_stage2, print_stage2_results
    
    print("=" * 60)
    print("  Stage 3: Safe Policy Deployment 테스트")
    print("=" * 60)
    
    # Stage 0
    print("\n[Stage 0] 데이터 생성")
    data, dgp = create_dataset(n_samples=3000, endogeneity=0.3, heterogeneity=0.5)
    feature_cols = dgp.get_feature_columns()
    
    # Stage 1
    bootstrap_dml, eval_dml = run_stage1(
        data, feature_cols, n_bootstrap=20, estimator_type="dml"
    )
    
    # Stage 2
    t_grid = np.linspace(0.05, 0.35, 31)
    stage2_results = run_stage2(
        data, feature_cols, bootstrap_dml, t_grid,
        alpha=0.1, synthesis_method="interaction",   # 논문 식(1)과 동일
    )
    print_stage2_results(stage2_results)
    
    # Stage 3
    all_results = run_stage3(
        data, stage2_results,
        baseline_t=0.15,
        lambda_values=[0.0, 0.3, 0.5, 0.7, 1.0],
    )
    print_stage3_results(all_results)
    
    print("\n✓ Stage 3 테스트 완료!")