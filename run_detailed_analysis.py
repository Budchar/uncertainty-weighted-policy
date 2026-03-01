"""
5.3 심층 분석용 스크립트: 상품별 상세 데이터 추출
==================================================

목적: 논문 5.3절 "왜 UW가 Naive보다 나은가?" 뒷받침 자료 생성

기존 run_experiment.py의 파이프라인(stage0→1→2→3)을 그대로 호출하되,
시나리오 평균이 아닌 **상품별(item-level) 상세 데이터**를 저장합니다.

생성되는 분석:
  분석 1: u(x) vs Naive 수익변화 scatter (위반 메커니즘)
  분석 2: 정책별 추천 마진 비교 (교정 시각화)
  분석 3: u(x) 구간별 성과 비교 (핵심 테이블)

사용법:
  python run_detailed_analysis.py
  
  ※ 프로젝트 루트에서 실행하세요 (src/ 폴더가 보이는 위치)
  ※ 대표 시나리오 3개만 돌리므로 기존 360회 대비 매우 빠릅니다.
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime

import sys
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.stage0_dgp import create_dataset
from src.stage1_estimate import run_stage1
from src.stage2_quantify import run_stage2
from src.stage3_deploy import (
    policy_naive_optimal,
    policy_uniform_conservative,
    policy_uncertainty_weighted,
    policy_threshold_based,
)


# =============================================================================
# 설정: 대표 시나리오 3개
# =============================================================================

ANALYSIS_SCENARIOS = [
    # (내생성, 샘플수, 이질성, 설명)
    (0.8, 1000, 0.8, "어려운 조건: UW 효과 극대화"),
    (0.5, 3000, 0.5, "중간 조건: 일반적 상황"),
    (0.1, 10000, 0.2, "쉬운 조건: UW 효과 최소"),
]

LAMBDA_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]
N_BOOTSTRAP = 20
BASELINE_T = 0.15
SEED = 42


# =============================================================================
# 핵심: 상품별 상세 데이터 추출
# =============================================================================

def extract_item_details(
    data: pd.DataFrame,
    stage2_results: dict,
    lambda_values: list,
    baseline_t: float = 0.15,
) -> pd.DataFrame:
    """상품별 상세 데이터를 DataFrame으로 추출
    
    각 상품(행)에 대해:
      - 불확실성 점수 (u_combined, u_stat, u_pos, u_flat)
      - 각 정책의 추천 마진 (naive_t, uw_t, oracle_t, baseline_t)
      - 각 정책의 수익 (naive_revenue, uw_revenue, oracle_revenue, baseline_revenue)
      - 위반 여부 (naive_violated, uw_violated)
    """
    from scipy.stats import rankdata                          # ← 추가
    
    true_alphas = data['true_alpha'].values
    base_demands = data['base_demand'].values
    true_optimal_t = data['true_optimal_margin'].values
    
    optimal_t = stage2_results['optimal_t']
    u_stat = stage2_results['u_statistical']
    u_pos = stage2_results['u_positivity']
    
    # ── 핵심 변경: percentile 정규화 + 두 합성 방법 ──
    n = len(u_stat)
    stat_p = (rankdata(u_stat) - 1) / max(n - 1, 1)          # [0, 1]
    pos_p  = (rankdata(u_pos)  - 1) / max(n - 1, 1)          # [0, 1]
    
    u_interact = (stat_p + pos_p + stat_p * pos_p) / 3.0     # interaction
    u_product  = stat_p * pos_p                                # 순수곱
    # ────────────────────────────────────────────────
    
    def revenue(t):
        return base_demands * np.exp(-true_alphas * t) * t
    
    baseline_rev = revenue(np.full(n, baseline_t))
    
    # 기본 데이터
    items = pd.DataFrame({
        # 불확실성 원천 (절대적 정규화 원본)
        'u_stat': u_stat,
        'u_pos': u_pos,
        # percentile 정규화 버전
        'stat_p': stat_p,
        'pos_p': pos_p,
        # 두 합성 방법
        'u_interact': u_interact,
        'u_product': u_product,
        
        # 마진 추천
        'oracle_t': true_optimal_t,
        'naive_t': optimal_t,
        'baseline_t': baseline_t,
        
        # 수익
        'oracle_revenue': revenue(true_optimal_t),
        'naive_revenue': revenue(optimal_t),
        'baseline_revenue': baseline_rev,
        
        # Naive 분석
        'naive_revenue_change': revenue(optimal_t) - baseline_rev,
        'naive_violated': revenue(optimal_t) < baseline_rev,
        'naive_t_error': optimal_t - true_optimal_t,
    })
    
    if 'category' in data.columns:
        items['category'] = data['category'].values
    
    # ── 핵심 변경: 두 합성 방법 × 확대된 λ ──
    for u_name, u_vals in [('uwI', u_interact), ('uwP', u_product)]:
        for lam in lambda_values:
            uw_t = policy_uncertainty_weighted(
                optimal_t, u_vals, baseline_t, lambda_param=lam
            )
            prefix = f'{u_name}{lam}'                # 예: uwI0.7, uwP3.0
            items[f'{prefix}_t'] = uw_t
            items[f'{prefix}_revenue'] = revenue(uw_t)
            items[f'{prefix}_violated'] = revenue(uw_t) < baseline_rev
            items[f'{prefix}_revenue_change'] = revenue(uw_t) - baseline_rev
    # ────────────────────────────────────────────────
    
    # Uniform 비교용 (변경 없음)
    for c in [0.3, 0.5, 0.7]:
        uni_t = policy_uniform_conservative(optimal_t, baseline_t, conservatism=c)
        prefix = f'uniform{c}'
        items[f'{prefix}_t'] = uni_t
        items[f'{prefix}_revenue'] = revenue(uni_t)
        items[f'{prefix}_violated'] = revenue(uni_t) < baseline_rev
        items[f'{prefix}_revenue_change'] = revenue(uni_t) - baseline_rev
    
    return items

# =============================================================================
# 분석 3: u(x) 구간별 성과 비교 (핵심 테이블)
# =============================================================================

def analyze_by_uncertainty_band(items: pd.DataFrame, lambda_val: float = 0.7) -> pd.DataFrame:
    """u(x) 구간별 Naive vs UW 성과 비교 — interaction 기준"""
    
    # u_combined → u_interact로 변경
    u_col = 'u_interact'
    uw_col = f'uwI{lambda_val}'
    
    bands = [
        ('Low (0~0.3)', 0, 0.3),
        ('Mid (0.3~0.7)', 0.3, 0.7),
        ('High (0.7~1.0)', 0.7, 1.01),
    ]
    
    rows = []
    for label, low, high in bands:
        mask = (items[u_col] >= low) & (items[u_col] < high)
        n = mask.sum()
        
        if n == 0:
            continue
        
        sub = items[mask]
        rows.append({
            'u(x) 구간': label,
            '상품 수': n,
            '비율': f'{n/len(items):.1%}',
            'Naive 위반율': f'{sub["naive_violated"].mean():.1%}',
            'Naive 평균수익변화': f'{sub["naive_revenue_change"].mean():.6f}',
            f'UW(λ={lambda_val}) 위반율': f'{sub[f"{uw_col}_violated"].mean():.1%}',
            f'UW(λ={lambda_val}) 평균수익변화': f'{sub[f"{uw_col}_revenue_change"].mean():.6f}',
            '평균|추정오차|': f'{sub["naive_t_error"].abs().mean():.4f}',
        })
    
    rows.append({
        'u(x) 구간': '전체',
        '상품 수': len(items),
        '비율': '100.0%',
        'Naive 위반율': f'{items["naive_violated"].mean():.1%}',
        'Naive 평균수익변화': f'{items["naive_revenue_change"].mean():.6f}',
        f'UW(λ={lambda_val}) 위반율': f'{items[f"{uw_col}_violated"].mean():.1%}',
        f'UW(λ={lambda_val}) 평균수익변화': f'{items[f"{uw_col}_revenue_change"].mean():.6f}',
        '평균|추정오차|': f'{items["naive_t_error"].abs().mean():.4f}',
    })
    
    return pd.DataFrame(rows)


# =============================================================================
# 추가 분석: 불확실성 원천별 기여도 (Ablation 준비)
# =============================================================================

def analyze_uncertainty_source_correlation(items: pd.DataFrame) -> dict:
    """각 불확실성 원천과 Naive 위반의 상관관계"""
    results = {}
    for src in ['u_stat', 'u_pos', 'stat_p', 'pos_p', 'u_interact', 'u_product']:
        corr = items[src].corr(items['naive_violated'].astype(float))
        results[src] = round(corr, 4)
    return results


# =============================================================================
# 메인 실행
# =============================================================================

def run_analysis():
    """전체 분석 실행"""
    
    output_dir = Path("./results/detailed_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_scenario_items = {}   # 시나리오별 상품 데이터
    all_band_tables = {}      # 시나리오별 구간 분석
    
    print("=" * 70)
    print("  5.3절 심층 분석: 상품별 상세 데이터 추출")
    print("=" * 70)
    
    for i, (endo, n_samples, hetero, desc) in enumerate(ANALYSIS_SCENARIOS):
        scenario_key = f"e{endo}_n{n_samples}_h{hetero}"
        
        print(f"\n{'='*60}")
        print(f"  시나리오 {i+1}/{len(ANALYSIS_SCENARIOS)}: {desc}")
        print(f"  endo={endo}, n={n_samples}, hetero={hetero}")
        print(f"{'='*60}")
        
        t0 = time.time()
        
        # --- Stage 0: 데이터 생성 ---
        data, dgp = create_dataset(
            n_samples=n_samples,
            endogeneity=endo,
            heterogeneity=hetero,
            random_seed=SEED,
        )
        feature_cols = dgp.get_feature_columns()
        
        # --- Stage 1: 추정 ---
        bootstrap_est, s1_eval = run_stage1(
            data, feature_cols,
            n_bootstrap=N_BOOTSTRAP,
            estimator_type='dml',
        )
        
        # --- Stage 2: 불확실성 ---
        t_grid = np.linspace(0.05, 0.35, 31)
        stage2_results = run_stage2(
            data, feature_cols, bootstrap_est, t_grid,
            alpha=0.1,
        )
        
        # --- 핵심: 상품별 상세 추출 ---
        items = extract_item_details(
            data, stage2_results, LAMBDA_VALUES, BASELINE_T
        )
        
        elapsed = time.time() - t0
        print(f"\n  ✓ 완료 ({elapsed:.0f}s) — {len(items)}개 상품 상세 데이터 추출")
        
        # --- 분석 3: 구간별 테이블 ---
        band_table = analyze_by_uncertainty_band(items, lambda_val=0.7)
        
        print(f"\n  📊 u(x) 구간별 분석:")
        print(band_table.to_string(index=False))
        
        # --- 불확실성-위반 상관 ---
        corr = analyze_uncertainty_source_correlation(items)
        print(f"\n  📈 불확실성 원천 vs Naive 위반 상관계수:")
        for src, val in corr.items():
            print(f"    {src}: {val}")
        
        # 저장
        all_scenario_items[scenario_key] = items
        all_band_tables[scenario_key] = band_table
        
        # CSV 저장
        items.to_csv(output_dir / f"items_{scenario_key}_{timestamp}.csv", index=False)
        band_table.to_csv(output_dir / f"bands_{scenario_key}_{timestamp}.csv", index=False)
    
    # --- 전체 결과 요약 ---
    print(f"\n{'='*70}")
    print(f"  전체 분석 완료")
    print(f"{'='*70}")
    
    for key, band_table in all_band_tables.items():
        print(f"\n[{key}]")
        print(band_table.to_string(index=False))
    
    print(f"\n결과 저장 위치: {output_dir}")
    print(f"파일 목록:")
    for f in sorted(output_dir.glob(f"*_{timestamp}*")):
        print(f"  {f.name}")
    
    return all_scenario_items, all_band_tables


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    all_items, all_bands = run_analysis()