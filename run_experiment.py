"""
체계적 비교 실험 (Systematic Comparison Experiment)
====================================================

3축 실험:
  - 내생성 강도: [0.1, 0.3, 0.5, 0.8]
  - 샘플 크기: [1000, 3000, 10000]
  - 이질성 강도: [0.2, 0.5, 0.8]

5개 방법 비교:
  1. naive_optimal: 추정 최적 그대로 적용
  2. uniform_conservative: 일률적 보수 (c=0.3, 0.5, 0.7)
  3. threshold_based: 불확실성 임계값 (th=0.3, 0.5, 0.7)
  4. uncertainty_weighted: 우리 방법 (λ sweep)
  5. oracle: 성능 상한

평가 지표:
  - mean_revenue: 평균 수익
  - safety_violation_rate: baseline보다 나빠지는 비율
  - mean_regret: oracle 대비 손실
  - revenue_at_risk: 하위 5% 수익 손실 (CVaR 유사)

사용법:
  python run_experiment.py                    # 전체 실험
  python run_experiment.py --quick            # 빠른 테스트 (축소 버전)
  python run_experiment.py --axis endogeneity # 내생성 축만 실험
  python run_experiment.py --resume           # 중단된 실험 이어서 실행
"""

import argparse
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product as cartesian_product

import sys
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.stage0_dgp import create_dataset
from src.stage1_estimate import run_stage1
from src.stage2_quantify import run_stage2
from src.stage3_deploy import run_stage3


# =============================================================================
# 실험 설정
# =============================================================================

FULL_CONFIG = {
    'endogeneity_levels': [0.1, 0.3, 0.5, 0.8],
    'sample_sizes': [1000, 3000, 10000],
    'heterogeneity_levels': [0.2, 0.5, 0.8],
    'lambda_values': [round(0.1 * i, 2) for i in range(0, 51)],
    'uniform_values': [round(0.05 * i, 2) for i in range(1, 20)],
    'threshold_values': [round(0.05 * i, 2) for i in range(1, 20)],
    'n_repeats': 10,
    'n_bootstrap': 20,
    'estimator_type': 'dml',
    'synthesis_methods': [
        ('interaction', None),    
    ],
}

QUICK_CONFIG = {
    'endogeneity_levels': [0.1, 0.5],
    'sample_sizes': [1000, 3000],
    'heterogeneity_levels': [0.2, 0.8],
    'lambda_values': [0.0, 0.5, 1.0, 3.0, 5.0],
    'uniform_values': [0.3, 0.5, 0.7],
    'threshold_values': [0.3, 0.5, 0.7],
    'n_repeats': 3,
    'n_bootstrap': 10,
    'estimator_type': 'dml',
    'synthesis_methods': [
        ('interaction', None),
    ],
}


# =============================================================================
# 단일 시나리오 실행
# =============================================================================

def run_single_scenario(
    endogeneity: float,
    n_samples: int,
    heterogeneity: float,
    lambda_values: list,
    uniform_values: list,
    threshold_values: list,
    n_bootstrap: int,
    estimator_type: str,
    seed: int,
    synthesis_method: str = "interaction",
    synthesis_weights: tuple = None,  
) -> dict:
    """단일 시나리오 (DGP 설정 1개 + 시드 1개) 실행
    
    Returns:
        모든 방법의 평가 결과를 담은 dict
    """
    
    # Stage 0: 데이터 생성
    data, dgp = create_dataset(
        n_samples=n_samples,
        endogeneity=endogeneity,
        heterogeneity=heterogeneity,
        random_seed=seed,
    )
    feature_cols = dgp.get_feature_columns()
    
    # Stage 1: Dose-response 추정
    bootstrap_est, s1_eval = run_stage1(
        data, feature_cols,
        n_bootstrap=n_bootstrap,
        estimator_type=estimator_type,
    )
    
    # Stage 2: 불확실성 정량화
    t_grid = np.linspace(0.05, 0.35, 31)
    stage2_results = run_stage2(
        data, feature_cols, bootstrap_est, t_grid,
        alpha=0.1, 
        synthesis_method=synthesis_method,
        synthesis_weights=synthesis_weights,
    )
    
    # Stage 3: 모든 방법 비교
    all_policy_results = run_stage3(
        data, stage2_results,
        baseline_t=0.15,
        lambda_values=lambda_values,
        uniform_values=uniform_values,
        threshold_values=threshold_values
    )

    # 결과 정리
    scenario_results = {
        'stage1_eval': s1_eval,
        'stage2_diagnostics': stage2_results.get('diagnostics', {}),
        'synthesis_method': synthesis_method,
        'synthesis_weights': list(synthesis_weights) if synthesis_weights else None,
        'policies': {},
    }
    
    for policy_name, metrics in all_policy_results.items():
        scenario_results['policies'][policy_name] = {
            'mean_revenue': metrics['mean_revenue'],
            'safety_violation_rate': metrics['safety_violation_rate'],
            'mean_regret': metrics['mean_regret'],
            'improvement_over_baseline': metrics['improvement_over_baseline'],
            'mean_revenue_shortfall': metrics.get('mean_revenue_shortfall', 0),
        }
    
    return scenario_results




# =============================================================================
# 전체 실험 실행
# =============================================================================
def _synthesis_label(method: str, weights: tuple = None) -> str:
    """합성 방법의 레이블 생성"""
    if method == "max":
        return "max"
    elif method == "stat_only":
        return "stat_only"
    elif method == "weighted":
        w = weights or (0.6, 0.2, 0.2)
        return f"w{w[0]}_{w[1]}_{w[2]}"
    return method

def run_experiment(config: dict, output_dir: str = "./results", resume: bool = False):
    """전체 3축 실험 실행"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 추정기 계열을 파일명에 넣는다. 같은 3축 설정을 추정기만 바꿔 돌릴 때
    # 타임스탬프만으로는 구분되지 않기 때문이다.
    est = config.get('estimator_type', 'dml')
    results_file = output_path / f"experiment_{est}_{timestamp}.json"
    # 체크포인트도 추정기별로 분리한다. 고정 이름을 쓰면 다른 추정기 실행이
    # 서로의 resume 상태를 덮어쓴다.
    checkpoint_file = output_path / f"checkpoint_{est}.json"
    
    # Resume: 기존 결과 로드
    completed = {}
    if resume and checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            completed = checkpoint.get('completed', {})
        print(f"  기존 체크포인트 로드: {len(completed)}개 시나리오 완료")
    
    # 시나리오 목록 생성
    scenarios = list(cartesian_product(
        config['endogeneity_levels'],
        config['sample_sizes'],
        config['heterogeneity_levels'],
    ))
    
    n_scenarios = len(scenarios)
    n_repeats = config['n_repeats']
    total_runs = n_scenarios * n_repeats
    
    print("=" * 70)
    print("  체계적 비교 실험 시작")
    print("=" * 70)
    print(f"  시나리오: {n_scenarios}개 (내생성 {len(config['endogeneity_levels'])} "
          f"× 샘플 {len(config['sample_sizes'])} "
          f"× 이질성 {len(config['heterogeneity_levels'])})")
    print(f"  반복: {n_repeats}회")
    print(f"  총 실행: {total_runs}회")
    print(f"  λ sweep: {config['lambda_values']}")
    print(f"  결과 저장: {results_file}")
    print("=" * 70)
    
    all_results = []
    run_count = 0
    start_time = time.time()
    
    for endo, n_samples, hetero in scenarios:
        scenario_key = f"e{endo}_n{n_samples}_h{hetero}"

        for repeat in range(n_repeats):
            seed = 42 + repeat * 1000

            # ★ Stage 0, 1은 한 번만 실행 (synthesis 방법 간 공유)
            data, dgp = create_dataset(
                n_samples=n_samples,
                endogeneity=endo,
                heterogeneity=hetero,
                random_seed=seed,
            )
            feature_cols = dgp.get_feature_columns()

            bootstrap_est, s1_eval = run_stage1(
                data, feature_cols,
                n_bootstrap=config['n_bootstrap'],
                estimator_type=config['estimator_type'],
            )

            t_grid = np.linspace(0.05, 0.35, 31)

            for syn_method, syn_weights in config['synthesis_methods']:
                syn_label = _synthesis_label(syn_method, syn_weights)
                run_key = f"{scenario_key}_r{repeat}_{syn_label}"

                if run_key in completed:
                    all_results.append(completed[run_key])
                    run_count += 1
                    continue

                run_count += 1
                print(f"\n[{run_count}/{total_runs}] "
                      f"endo={endo}, n={n_samples}, hetero={hetero}, "
                      f"repeat={repeat+1}/{n_repeats}, "
                      f"synthesis={syn_label}")

                try:
                    t0 = time.time()

                    # Stage 2
                    stage2_results = run_stage2(
                        data, feature_cols, bootstrap_est, t_grid,
                        alpha=0.1,
                        synthesis_method=syn_method,
                        synthesis_weights=syn_weights,
                    )

                    # Stage 3
                    all_policy_results = run_stage3(
                        data, stage2_results,
                        baseline_t=0.15,
                        lambda_values=config['lambda_values'],
                        uniform_values=config['uniform_values'],
                        threshold_values=config['threshold_values'],
                    )

                    elapsed = time.time() - t0

                    result = {
                        'stage1_eval': s1_eval,
                        'stage2_diagnostics': stage2_results.get('diagnostics', {}),
                        'synthesis_method': syn_method,
                        'synthesis_weights': list(syn_weights) if syn_weights else None,
                        'synthesis_label': syn_label,
                        'policies': {},
                        'scenario': {
                            'endogeneity': endo,
                            'n_samples': n_samples,
                            'heterogeneity': hetero,
                            'repeat': repeat,
                            'seed': seed,
                            'elapsed_seconds': round(elapsed, 1),
                        },
                    }

                    for policy_name, metrics in all_policy_results.items():
                        result['policies'][policy_name] = {
                            'mean_revenue': metrics['mean_revenue'],
                            'safety_violation_rate': metrics['safety_violation_rate'],
                            'mean_regret': metrics['mean_regret'],
                            'improvement_over_baseline': metrics['improvement_over_baseline'],
                            'mean_revenue_shortfall': metrics.get('mean_revenue_shortfall', 0),
                        }

                    all_results.append(result)
                    completed[run_key] = result
                    _save_checkpoint(checkpoint_file, completed, config)
                    _print_progress(result, elapsed, run_count, total_runs, start_time)

                except Exception as e:
                    print(f"  ❌ 에러: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    # 최종 결과 저장
    final_output = {
        'config': config,
        'timestamp': timestamp,
        'total_elapsed': round(time.time() - start_time, 1),
        'results': all_results,
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_output, f, indent=2, default=str)
    
    print(f"\n{'=' * 70}")
    print(f"  실험 완료! 결과: {results_file}")
    print(f"  총 소요시간: {(time.time() - start_time)/60:.1f}분")
    print(f"{'=' * 70}")
    
    # 요약 테이블 생성
    summary_df = create_summary_table(all_results)
    summary_file = output_path / f"summary_{est}_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"  요약 테이블: {summary_file}")
    
    return all_results, summary_df


def _save_checkpoint(checkpoint_file, completed, config):
    """체크포인트 저장"""
    checkpoint = {
        'config': config,
        'completed': completed,
        'last_saved': datetime.now().isoformat(),
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, default=str)


def _print_progress(result, elapsed, run_count, total_runs, start_time):
    """진행 상황 출력"""
    # 핵심 지표만 간단히 출력
    policies = result['policies']
    
    naive_viol = policies.get('naive_optimal', {}).get('safety_violation_rate', -1)
    
    # 가장 좋은 uw 찾기
    best_uw_name = None
    best_uw_viol = 1.0
    for name, metrics in policies.items():
        if name.startswith('uw_lambda') and name != 'uw_lambda0.0':
            if metrics['safety_violation_rate'] < best_uw_viol:
                best_uw_viol = metrics['safety_violation_rate']
                best_uw_name = name
    
    best_uw_improve = policies.get(best_uw_name, {}).get('improvement_over_baseline', 0)
    
    total_elapsed = time.time() - start_time
    avg_per_run = total_elapsed / run_count
    remaining = avg_per_run * (total_runs - run_count)
    
    print(f"  ✓ {elapsed:.0f}s | "
          f"Naive위반={naive_viol:.1%} → "
          f"UW위반={best_uw_viol:.1%} "
          f"(개선={best_uw_improve:+.1f}%) | "
          f"남은시간≈{remaining/60:.0f}분")


# =============================================================================
# 결과 요약
# =============================================================================

def create_summary_table(all_results: list) -> pd.DataFrame:
    """실험 결과를 요약 테이블로 변환"""
    
    rows = []
    for result in all_results:
        scenario = result['scenario']
        
        for policy_name, metrics in result['policies'].items():
            rows.append({
                'endogeneity': scenario['endogeneity'],
                'n_samples': scenario['n_samples'],
                'heterogeneity': scenario['heterogeneity'],
                'repeat': scenario['repeat'],
                'seed': scenario['seed'],
                'policy': policy_name,
                'mean_revenue': metrics['mean_revenue'],
                'safety_violation_rate': metrics['safety_violation_rate'],
                'mean_regret': metrics['mean_regret'],
                'improvement_over_baseline': metrics['improvement_over_baseline'],
                'mean_revenue_shortfall': metrics.get('mean_revenue_shortfall', 0),
                # Stage 1 정보
                's1_optimal_t_mae': result['stage1_eval'].get('optimal_t_mae', None),
                's1_violation_rate': result['stage1_eval'].get('safety_violation_rate', None),
            })
    
    return pd.DataFrame(rows)


def print_summary(summary_df: pd.DataFrame):
    """핵심 결과 요약 출력"""
    
    print("\n" + "=" * 80)
    print("  실험 결과 요약")
    print("=" * 80)
    
    # 1. 전체 평균: 방법별 비교
    print("\n[1] 전체 평균 — 방법별 비교")
    print("-" * 70)
    
    method_summary = summary_df.groupby('policy').agg({
        'improvement_over_baseline': 'mean',
        'safety_violation_rate': 'mean',
        'mean_regret': 'mean',
    }).round(4)
    
    method_summary = method_summary.sort_values('safety_violation_rate')
    
    print(f"{'방법':<25} | {'개선율':>8} | {'위반율':>8} | {'Regret':>8}")
    print(f"{'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for name, row in method_summary.iterrows():
        print(f"{name:<25} | {row['improvement_over_baseline']:>+7.2f}% | "
              f"{row['safety_violation_rate']:>7.1%} | "
              f"{row['mean_regret']:>8.5f}")
    
    # 2. 내생성별: 우리 방법의 효과
    print("\n[2] 내생성별 — UW(λ=0.7) vs Naive")
    print("-" * 70)
    
    for endo in sorted(summary_df['endogeneity'].unique()):
        sub = summary_df[summary_df['endogeneity'] == endo]
        
        naive = sub[sub['policy'] == 'naive_optimal']
        uw07 = sub[sub['policy'] == 'uw_lambda0.7']
        
        if len(naive) > 0 and len(uw07) > 0:
            print(f"  내생성={endo}: "
                  f"Naive 위반={naive['safety_violation_rate'].mean():.1%} → "
                  f"UW 위반={uw07['safety_violation_rate'].mean():.1%} | "
                  f"UW 개선={uw07['improvement_over_baseline'].mean():+.2f}%")
    
    # 3. 샘플 크기별
    print("\n[3] 샘플 크기별 — UW(λ=0.7) vs Naive")
    print("-" * 70)
    
    for n in sorted(summary_df['n_samples'].unique()):
        sub = summary_df[summary_df['n_samples'] == n]
        
        naive = sub[sub['policy'] == 'naive_optimal']
        uw07 = sub[sub['policy'] == 'uw_lambda0.7']
        
        if len(naive) > 0 and len(uw07) > 0:
            print(f"  n={n:>5}: "
                  f"Naive 위반={naive['safety_violation_rate'].mean():.1%} → "
                  f"UW 위반={uw07['safety_violation_rate'].mean():.1%} | "
                  f"UW 개선={uw07['improvement_over_baseline'].mean():+.2f}%")
    
    # 4. 이질성별
    print("\n[4] 이질성별 — UW(λ=0.7) vs Naive")
    print("-" * 70)
    
    for h in sorted(summary_df['heterogeneity'].unique()):
        sub = summary_df[summary_df['heterogeneity'] == h]
        
        naive = sub[sub['policy'] == 'naive_optimal']
        uw07 = sub[sub['policy'] == 'uw_lambda0.7']
        
        if len(naive) > 0 and len(uw07) > 0:
            print(f"  이질성={h}: "
                  f"Naive 위반={naive['safety_violation_rate'].mean():.1%} → "
                  f"UW 위반={uw07['safety_violation_rate'].mean():.1%} | "
                  f"UW 개선={uw07['improvement_over_baseline'].mean():+.2f}%")
    
    # 5. λ tradeoff curve
    print("\n[5] λ별 Safety-Revenue Tradeoff (전체 평균)")
    print("-" * 70)
    
    uw_policies = summary_df[summary_df['policy'].str.startswith('uw_lambda')]
    if len(uw_policies) > 0:
        lambda_summary = uw_policies.groupby('policy').agg({
            'improvement_over_baseline': 'mean',
            'safety_violation_rate': 'mean',
        }).sort_values('safety_violation_rate')
        
        print(f"{'λ':<15} | {'개선율':>8} | {'위반율':>8}")
        print(f"{'-'*15}-+-{'-'*8}-+-{'-'*8}")
        for name, row in lambda_summary.iterrows():
            lam_val = name.replace('uw_lambda', 'λ=')
            print(f"{lam_val:<15} | {row['improvement_over_baseline']:>+7.2f}% | "
                  f"{row['safety_violation_rate']:>7.1%}")
    
    print(f"\n{'=' * 80}")


# =============================================================================
# 단일 축 실험 (디버깅/빠른 확인용)
# =============================================================================

def run_single_axis(axis: str, config: dict, output_dir: str = "./results"):
    """단일 축만 실험 (다른 축은 기본값 고정)"""
    
    defaults = {
        'endogeneity': 0.3,
        'n_samples': 3000,
        'heterogeneity': 0.5,
    }
    
    if axis == 'endogeneity':
        varying = config['endogeneity_levels']
    elif axis == 'sample_size':
        varying = config['sample_sizes']
    elif axis == 'heterogeneity':
        varying = config['heterogeneity_levels']
    else:
        raise ValueError(f"Unknown axis: {axis}. Use: endogeneity, sample_size, heterogeneity")
    
    print(f"\n단일 축 실험: {axis}")
    print(f"  변동: {varying}")
    print(f"  고정: {', '.join(f'{k}={v}' for k, v in defaults.items() if k != axis.replace('sample_size', 'n_samples'))}")
    
    all_results = []
    
    for val in varying:
        for repeat in range(config['n_repeats']):
            params = defaults.copy()
            if axis == 'endogeneity':
                params['endogeneity'] = val
            elif axis == 'sample_size':
                params['n_samples'] = val
            elif axis == 'heterogeneity':
                params['heterogeneity'] = val
            
            seed = 42 + repeat * 1000
            
            print(f"\n  [{axis}={val}, repeat={repeat+1}]")
            
            try:
                result = run_single_scenario(
                    endogeneity=params['endogeneity'],
                    n_samples=params['n_samples'],
                    heterogeneity=params['heterogeneity'],
                    lambda_values=config['lambda_values'],
                    n_bootstrap=config['n_bootstrap'],
                    estimator_type=config['estimator_type'],
                    seed=seed,
                    uniform_values=config['uniform_values'],
                    threshold_values=config['threshold_values'],
                )
                result['scenario'] = {
                    'endogeneity': params['endogeneity'],
                    'n_samples': params['n_samples'],
                    'heterogeneity': params['heterogeneity'],
                    'repeat': repeat,
                    'seed': seed,
                }
                all_results.append(result)
            except Exception as e:
                print(f"  ❌ 에러: {e}")
    
    summary_df = create_summary_table(all_results)
    print_summary(summary_df)
    
    return all_results, summary_df


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="체계적 비교 실험")
    parser.add_argument('--quick', action='store_true',
                        help='빠른 테스트 (축소 버전)')
    parser.add_argument('--axis', type=str, default=None,
                        choices=['endogeneity', 'sample_size', 'heterogeneity'],
                        help='단일 축만 실험')
    parser.add_argument('--resume', action='store_true',
                        help='중단된 실험 이어서 실행')
    parser.add_argument('--output', type=str, default='./results',
                        help='결과 저장 디렉토리')
    parser.add_argument('--estimator', type=str, default=None,
                        choices=['dml', 'odml', 'gps'],
                        help='추정기 계열 (odml = 직교화를 실제로 적용하는 DML)')
    parser.add_argument('--repeats', type=int, default=None,
                        help='반복 횟수 오버라이드')
    
    args = parser.parse_args()
    
    # 설정 선택
    config = QUICK_CONFIG.copy() if args.quick else FULL_CONFIG.copy()
    
    if args.repeats is not None:
        config['n_repeats'] = args.repeats

    if args.estimator is not None:
        config['estimator_type'] = args.estimator

    # 실행
    if args.axis:
        results, summary = run_single_axis(args.axis, config, args.output)
    else:
        results, summary = run_experiment(config, args.output, resume=args.resume)
    
    # 요약 출력
    print_summary(summary)