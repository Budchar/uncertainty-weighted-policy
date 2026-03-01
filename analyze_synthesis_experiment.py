"""
Synthesis 방법 비교 실험 분석 스크립트
=====================================

사용법:
  python analyze_synthesis_experiment.py --json results/experiment_XXXXXXXX.json

또는 summary CSV가 synthesis 컬럼을 포함하고 있으면:
  python analyze_synthesis_experiment.py --csv results/summary_XXXXXXXX.csv
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d


# =============================================================================
# 1. 데이터 로드
# =============================================================================

def load_from_json(json_path: str) -> pd.DataFrame:
    """JSON 실험 결과에서 synthesis 포함 summary table 생성"""
    with open(json_path) as f:
        data = json.load(f)
    
    rows = []
    for result in data['results']:
        scenario = result['scenario']
        syn_label = result.get('synthesis_label', 'max')
        for policy_name, metrics in result['policies'].items():
            rows.append({
                'endogeneity': scenario['endogeneity'],
                'n_samples': scenario['n_samples'],
                'heterogeneity': scenario['heterogeneity'],
                'repeat': scenario['repeat'],
                'seed': scenario['seed'],
                'synthesis': syn_label,
                'policy': policy_name,
                'mean_revenue': metrics['mean_revenue'],
                'safety_violation_rate': metrics['safety_violation_rate'],
                'mean_regret': metrics['mean_regret'],
                'improvement_over_baseline': metrics['improvement_over_baseline'],
                'mean_revenue_shortfall': metrics.get('mean_revenue_shortfall', 0),
                's1_optimal_t_mae': result['stage1_eval'].get('optimal_t_mae', None),
            })
    
    df = pd.DataFrame(rows)
    print(f"로드 완료: {len(df)} rows")
    print(f"  synthesis: {sorted(df['synthesis'].unique())}")
    print(f"  policies: {sorted(df['policy'].unique())}")
    print(f"  scenarios: {len(df.groupby(['endogeneity','n_samples','heterogeneity']))}")
    print(f"  repeats: {sorted(df['repeat'].unique())}")
    return df


def load_from_csv(csv_path: str) -> pd.DataFrame:
    """CSV에 synthesis 컬럼이 있으면 그대로 사용"""
    df = pd.read_csv(csv_path)
    if 'synthesis' not in df.columns:
        raise ValueError("CSV에 'synthesis' 컬럼이 없습니다. --json 옵션을 사용하세요.")
    print(f"로드 완료: {len(df)} rows")
    return df


# =============================================================================
# 2. 분석 1: Synthesis 방법별 전체 평균 비교
# =============================================================================

def analysis_1_overall(df: pd.DataFrame):
    """각 synthesis 방법별로 UW의 전체 평균 성과 비교"""
    
    print(f"\n{'='*80}")
    print("분석 1: Synthesis 방법별 UW 전체 평균")
    print(f"{'='*80}")
    
    # UW 정책만 추출
    uw = df[df['policy'].str.startswith('uw_lambda')]
    
    # synthesis × policy 별 평균
    comp = uw.groupby(['synthesis', 'policy']).agg({
        'improvement_over_baseline': 'mean',
        'safety_violation_rate': 'mean',
    }).round(4)
    
    for syn in sorted(df['synthesis'].unique()):
        print(f"\n  --- {syn} ---")
        sub = comp.loc[syn] if syn in comp.index.get_level_values(0) else pd.DataFrame()
        if len(sub) > 0:
            for pol, row in sub.iterrows():
                lam = pol.replace('uw_lambda', 'λ=')
                print(f"    {lam:>8s}: rev={row['improvement_over_baseline']:+.3f}%  "
                      f"viol={row['safety_violation_rate']:.1%}")
    
    # Naive와 Uniform은 synthesis에 영향 안 받으므로 참고용 출력
    print(f"\n  --- 참고: synthesis 무관 정책 (첫 번째 synthesis 기준) ---")
    first_syn = sorted(df['synthesis'].unique())[0]
    ref = df[df['synthesis'] == first_syn]
    for pol in ['naive_optimal', 'uniform_c0.3', 'uniform_c0.5', 'uniform_c0.7', 'oracle']:
        sub = ref[ref['policy'] == pol]
        if len(sub) > 0:
            print(f"    {pol:>20s}: rev={sub['improvement_over_baseline'].mean():+.3f}%  "
                  f"viol={sub['safety_violation_rate'].mean():.1%}")


# =============================================================================
# 3. 분석 2: 같은 위반율에서 수익 비교 (보간)
# =============================================================================

def analysis_2_matched_violation(df: pd.DataFrame):
    """각 시나리오에서 synthesis 방법 간 tradeoff curve를 비교"""
    
    print(f"\n{'='*80}")
    print("분석 2: 같은 위반율에서 수익 비교 (synthesis 간 보간)")
    print(f"{'='*80}")
    
    synths = sorted(df['synthesis'].unique())
    scenarios = df.groupby(['endogeneity', 'n_samples', 'heterogeneity']).size().reset_index()[
        ['endogeneity', 'n_samples', 'heterogeneity']
    ].values.tolist()
    
    # 타겟 위반율들
    target_viols = [0.10, 0.12, 0.15, 0.20]
    
    results = []
    for endo, n, h in scenarios:
        for syn in synths:
            cond = (
                (df['endogeneity'] == endo) & 
                (df['n_samples'] == n) & 
                (df['heterogeneity'] == h) & 
                (df['synthesis'] == syn) &
                (df['policy'].str.startswith('uw_lambda'))
            )
            sub = df[cond].groupby('policy').agg({
                'improvement_over_baseline': 'mean',
                'safety_violation_rate': 'mean',
            }).reset_index()
            
            if len(sub) < 3:
                continue
            
            # 위반율 기준 정렬 후 보간
            sub_sorted = sub.sort_values('safety_violation_rate')
            try:
                interp = interp1d(
                    sub_sorted['safety_violation_rate'].values,
                    sub_sorted['improvement_over_baseline'].values,
                    kind='linear', fill_value='extrapolate'
                )
                
                for tv in target_viols:
                    rev_at_tv = float(interp(tv))
                    results.append({
                        'endogeneity': endo, 'n_samples': n, 'heterogeneity': h,
                        'synthesis': syn,
                        'target_viol': tv,
                        'revenue_at_target': rev_at_tv,
                    })
            except Exception:
                continue
    
    rdf = pd.DataFrame(results)
    
    if len(rdf) == 0:
        print("  보간 결과 없음 (lambda 값이 부족할 수 있음)")
        return
    
    # 타겟 위반율별 synthesis 비교
    for tv in target_viols:
        print(f"\n  위반율 = {tv:.0%} 에서의 수익:")
        sub = rdf[rdf['target_viol'] == tv]
        comp = sub.groupby('synthesis')['revenue_at_target'].mean()
        for syn, rev in comp.items():
            print(f"    {syn:>15s}: rev={rev:+.3f}%")
        
        if len(comp) >= 2:
            best = comp.idxmax()
            worst = comp.idxmin()
            print(f"    → {best}이 {worst} 대비 {comp[best]-comp[worst]:+.3f}%p 우위")


# =============================================================================
# 4. 분석 3: 시나리오 난이도별 synthesis 효과
# =============================================================================

def analysis_3_by_difficulty(df: pd.DataFrame):
    """내생성/샘플크기별로 synthesis 방법의 효과가 다른지 확인"""
    
    print(f"\n{'='*80}")
    print("분석 3: 시나리오 난이도별 synthesis 효과")
    print(f"{'='*80}")
    
    synths = sorted(df['synthesis'].unique())
    
    # λ=0.7 있으면 그걸로, 없으면 가장 가까운 것
    uw_policies = sorted(df[df['policy'].str.startswith('uw_lambda')]['policy'].unique())
    # λ=0.7 > 0.5 > 1.0 순으로 선택
    target_pol = None
    for candidate in ['uw_lambda0.7', 'uw_lambda0.5', 'uw_lambda1.0']:
        if candidate in uw_policies:
            target_pol = candidate
            break
    
    if target_pol is None:
        print("  UW 정책을 찾을 수 없습니다.")
        return
    
    lam_label = target_pol.replace('uw_lambda', 'λ=')
    print(f"  비교 정책: UW({lam_label})")
    
    uw = df[df['policy'] == target_pol]
    
    # 내생성별
    print(f"\n  --- 내생성별 ---")
    print(f"  {'endo':>6s}", end="")
    for syn in synths:
        print(f" | {syn:>15s} rev  {syn:>15s} viol", end="")
    print()
    
    for endo in sorted(uw['endogeneity'].unique()):
        print(f"  {endo:>6.1f}", end="")
        for syn in synths:
            sub = uw[(uw['endogeneity'] == endo) & (uw['synthesis'] == syn)]
            rev = sub['improvement_over_baseline'].mean()
            viol = sub['safety_violation_rate'].mean()
            print(f" | {rev:>15.3f}%  {viol:>14.1%}", end="")
        print()
    
    # 샘플크기별
    print(f"\n  --- 샘플크기별 ---")
    print(f"  {'n':>6s}", end="")
    for syn in synths:
        print(f" | {syn:>15s} rev  {syn:>15s} viol", end="")
    print()
    
    for n in sorted(uw['n_samples'].unique()):
        print(f"  {n:>6d}", end="")
        for syn in synths:
            sub = uw[(uw['n_samples'] == n) & (uw['synthesis'] == syn)]
            rev = sub['improvement_over_baseline'].mean()
            viol = sub['safety_violation_rate'].mean()
            print(f" | {rev:>15.3f}%  {viol:>14.1%}", end="")
        print()


# =============================================================================
# 5. 분석 4: Synthesis 방법 간 승패 집계
# =============================================================================

def analysis_4_win_rate(df: pd.DataFrame):
    """36개 시나리오 각각에서 어떤 synthesis가 더 나은지 집계"""
    
    print(f"\n{'='*80}")
    print("분석 4: 시나리오별 synthesis 승패 (UW 정책 기준)")
    print(f"{'='*80}")
    
    synths = sorted(df['synthesis'].unique())
    if len(synths) < 2:
        print("  synthesis 방법이 2개 미만이라 비교 불가")
        return
    
    # λ 선택
    uw_policies = sorted(df[df['policy'].str.startswith('uw_lambda')]['policy'].unique())
    target_pol = None
    for candidate in ['uw_lambda0.7', 'uw_lambda0.5', 'uw_lambda1.0']:
        if candidate in uw_policies:
            target_pol = candidate
            break
    
    lam_label = target_pol.replace('uw_lambda', 'λ=')
    print(f"  비교 정책: UW({lam_label})")
    
    uw = df[df['policy'] == target_pol]
    
    # 시나리오별 평균
    avg = uw.groupby(['endogeneity', 'n_samples', 'heterogeneity', 'synthesis']).agg({
        'improvement_over_baseline': 'mean',
        'safety_violation_rate': 'mean',
    }).reset_index()
    
    # 모든 synthesis 쌍에 대해 승패 집계
    for i, syn_a in enumerate(synths):
        for syn_b in synths[i+1:]:
            print(f"\n  --- {syn_a} vs {syn_b} ---")
            
            a_rev_wins = 0
            b_rev_wins = 0
            a_viol_wins = 0
            b_viol_wins = 0
            total = 0
            
            scenarios = avg.groupby(['endogeneity', 'n_samples', 'heterogeneity']).size().reset_index()
            
            for _, row in scenarios.iterrows():
                endo, n, h = row['endogeneity'], row['n_samples'], row['heterogeneity']
                a = avg[(avg['endogeneity']==endo) & (avg['n_samples']==n) & 
                        (avg['heterogeneity']==h) & (avg['synthesis']==syn_a)]
                b = avg[(avg['endogeneity']==endo) & (avg['n_samples']==n) & 
                        (avg['heterogeneity']==h) & (avg['synthesis']==syn_b)]
                
                if len(a) == 0 or len(b) == 0:
                    continue
                
                total += 1
                a_rev = a.iloc[0]['improvement_over_baseline']
                b_rev = b.iloc[0]['improvement_over_baseline']
                a_viol = a.iloc[0]['safety_violation_rate']
                b_viol = b.iloc[0]['safety_violation_rate']
                
                if a_rev > b_rev:
                    a_rev_wins += 1
                else:
                    b_rev_wins += 1
                
                if a_viol < b_viol:
                    a_viol_wins += 1
                else:
                    b_viol_wins += 1
            
            print(f"    수익 승: {syn_a} {a_rev_wins}/{total}, {syn_b} {b_rev_wins}/{total}")
            print(f"    위반 승: {syn_a} {a_viol_wins}/{total}, {syn_b} {b_viol_wins}/{total}")


# =============================================================================
# 6. 그래프: Tradeoff curve 비교
# =============================================================================

def plot_tradeoff_curves(df: pd.DataFrame, output_dir: str = "."):
    """Synthesis 방법별 safety-revenue tradeoff curve"""
    
    synths = sorted(df['synthesis'].unique())
    colors = {'max': '#e03131', 'stat_only': '#1971c2', 'w0.6_0.2_0.2': '#2f9e44',
              'w0.8_0.1_0.1': '#e8590c'}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # (a) 전체 평균 tradeoff curve
    ax = axes[0]
    for syn in synths:
        uw = df[(df['synthesis'] == syn) & (df['policy'].str.startswith('uw_lambda'))]
        avg = uw.groupby('policy').agg({
            'improvement_over_baseline': 'mean',
            'safety_violation_rate': 'mean',
        }).sort_values('safety_violation_rate')
        
        color = colors.get(syn, '#868e96')
        ax.plot(avg['safety_violation_rate'] * 100, avg['improvement_over_baseline'],
                'o-', color=color, lw=2, ms=6, label=syn)
    
    # Uniform 참고선 (synthesis 무관)
    first_syn = synths[0]
    for c in ['0.3', '0.5', '0.7']:
        uni = df[(df['synthesis'] == first_syn) & (df['policy'] == f'uniform_c{c}')]
        if len(uni) > 0:
            ax.scatter(uni['safety_violation_rate'].mean() * 100,
                      uni['improvement_over_baseline'].mean(),
                      marker='D', s=60, color='#868e96', zorder=5)
            ax.annotate(f'Uni({c})', 
                       (uni['safety_violation_rate'].mean() * 100,
                        uni['improvement_over_baseline'].mean()),
                       textcoords='offset points', xytext=(5, 5), fontsize=8, color='#868e96')
    
    ax.set_xlabel('Safety violation rate (%)')
    ax.set_ylabel('Revenue improvement (%)')
    ax.set_title('(a) Overall tradeoff curve by synthesis method')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # (b) 고혼동 시나리오만
    ax = axes[1]
    high_endo = df['endogeneity'].max()
    for syn in synths:
        uw = df[(df['synthesis'] == syn) & 
                (df['policy'].str.startswith('uw_lambda')) &
                (df['endogeneity'] == high_endo)]
        avg = uw.groupby('policy').agg({
            'improvement_over_baseline': 'mean',
            'safety_violation_rate': 'mean',
        }).sort_values('safety_violation_rate')
        
        color = colors.get(syn, '#868e96')
        ax.plot(avg['safety_violation_rate'] * 100, avg['improvement_over_baseline'],
                'o-', color=color, lw=2, ms=6, label=syn)
    
    ax.set_xlabel('Safety violation rate (%)')
    ax.set_ylabel('Revenue improvement (%)')
    ax.set_title(f'(b) High confounding (γ={high_endo}) only')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.suptitle('Synthesis method comparison: Safety-Revenue Tradeoff',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = Path(output_dir) / 'fig_synthesis_comparison.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\n  그래프 저장: {out_path}")


# =============================================================================
# 7. 분석 5: 최종 추천 — 어떤 synthesis를 쓸지
# =============================================================================

def analysis_5_recommendation(df: pd.DataFrame):
    """최종 추천 요약"""
    
    print(f"\n{'='*80}")
    print("분석 5: 최종 추천")
    print(f"{'='*80}")
    
    synths = sorted(df['synthesis'].unique())
    
    # λ 선택
    uw_policies = sorted(df[df['policy'].str.startswith('uw_lambda')]['policy'].unique())
    
    print(f"\n  전체 평균 비교 (모든 λ):")
    print(f"  {'synthesis':>15s} | {'best_rev':>10s} | {'best_viol':>10s} | {'best_λ_rev':>12s} | {'best_λ_viol':>12s}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")
    
    for syn in synths:
        uw = df[(df['synthesis'] == syn) & (df['policy'].str.startswith('uw_lambda'))]
        avg = uw.groupby('policy').agg({
            'improvement_over_baseline': 'mean',
            'safety_violation_rate': 'mean',
        })
        
        best_rev_pol = avg['improvement_over_baseline'].idxmax()
        best_viol_pol = avg['safety_violation_rate'].idxmin()
        
        best_rev = avg.loc[best_rev_pol, 'improvement_over_baseline']
        best_viol = avg.loc[best_viol_pol, 'safety_violation_rate']
        
        print(f"  {syn:>15s} | {best_rev:>+9.3f}% | {best_viol:>9.1%} | "
              f"{best_rev_pol.replace('uw_lambda','λ='):>12s} | "
              f"{best_viol_pol.replace('uw_lambda','λ='):>12s}")
    
    print(f"\n  해석:")
    print(f"  - max: u_pos, u_flat의 노이즈로 인해 불필요한 보수성 추가 → 수익 손실")
    print(f"  - stat_only: u_stat만 사용하여 노이즈 제거 → 같은 λ에서 수익 개선")
    print(f"  - weighted: 중간 성과, u_pos/u_flat의 부분적 기여 반영")


# =============================================================================
# 메인
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Synthesis 방법 비교 분석")
    parser.add_argument('--json', type=str, help='실험 JSON 파일 경로')
    parser.add_argument('--csv', type=str, help='Summary CSV 파일 경로 (synthesis 컬럼 필요)')
    parser.add_argument('--output', type=str, default='.', help='그래프 저장 디렉토리')
    args = parser.parse_args()
    
    if args.json:
        df = load_from_json(args.json)
    elif args.csv:
        df = load_from_csv(args.csv)
    else:
        # 현재 디렉토리에서 가장 최근 json 찾기
        json_files = sorted(Path('./results').glob('experiment_*.json'), reverse=True)
        if json_files:
            print(f"자동 감지: {json_files[0]}")
            df = load_from_json(str(json_files[0]))
        else:
            print("사용법: python analyze_synthesis_experiment.py --json <경로>")
            return
    
    # synthesis 컬럼 확인
    if 'synthesis' not in df.columns:
        print("ERROR: synthesis 컬럼이 없습니다. 실험을 수정된 코드로 다시 돌려주세요.")
        return
    
    # 분석 실행
    analysis_1_overall(df)
    analysis_2_matched_violation(df)
    analysis_3_by_difficulty(df)
    analysis_4_win_rate(df)
    analysis_5_recommendation(df)
    
    # 그래프
    plot_tradeoff_curves(df, args.output)
    
    # summary CSV 저장 (synthesis 포함)
    out_csv = Path(args.output) / 'summary_with_synthesis.csv'
    df.to_csv(out_csv, index=False)
    print(f"\n  Summary CSV 저장: {out_csv}")


if __name__ == "__main__":
    main()