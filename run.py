"""
전체 파이프라인 실행 (캐시 지원)
=================================

사용법:
    # 전체 실행 (캐시 자동 사용)
    python run_pipeline.py
    
    # 특정 stage부터 재실행
    python run_pipeline.py --from-stage 2
    
    # 캐시 무시하고 전체 재실행
    python run_pipeline.py --force
    
    # 캐시 목록 확인
    python run_pipeline.py --list-cache
"""

import argparse
import numpy as np
from pathlib import Path
import sys

_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.cache_utils import StageCache
from src.stage0_dgp import create_dataset
from src.stage1_estimate import run_stage1, print_evaluation
from src.stage2_quantify import run_stage2, print_stage2_results
from src.stage3_deploy import run_stage3, print_stage3_results


def run_pipeline(
    n_samples: int = 3000,
    endogeneity: float = 0.3,
    heterogeneity: float = 0.5,
    n_bootstrap: int = 20,
    estimator_type: str = "dml",
    from_stage: int = 0,
    force: bool = False,
    cache_dir: str = "./cache",
):
    """전체 파이프라인 실행"""
    
    cache = StageCache(cache_dir)
    
    # 공통 파라미터 (캐시 키)
    base_params = {
        'n_samples': n_samples,
        'endogeneity': endogeneity,
        'heterogeneity': heterogeneity,
    }
    
    print("=" * 60)
    print("  Safe Policy Deployment Pipeline")
    print("=" * 60)
    print(f"  n_samples={n_samples}, endogeneity={endogeneity}")
    print(f"  estimator={estimator_type}, bootstrap={n_bootstrap}")
    print(f"  cache_dir={cache_dir}")
    print("=" * 60)
    
    # ===== Stage 0: DGP =====
    s0_params = {**base_params}
    force_s0 = force or from_stage <= 0
    
    def _run_s0():
        print("\n[Stage 0] 데이터 생성")
        data, dgp = create_dataset(
            n_samples=n_samples,
            endogeneity=endogeneity,
            heterogeneity=heterogeneity,
        )
        feature_cols = dgp.get_feature_columns()
        return {'data': data, 'feature_cols': feature_cols}
    
    s0 = cache.get_or_run("stage0", _run_s0, s0_params, force_rerun=force_s0)
    data = s0['data']
    feature_cols = s0['feature_cols']
    print(f"  데이터: {len(data)} samples")
    
    # ===== Stage 1: Estimation =====
    s1_params = {**base_params, 'n_bootstrap': n_bootstrap, 'estimator': estimator_type}
    force_s1 = force or from_stage <= 1
    
    def _run_s1():
        bootstrap_est, eval_results = run_stage1(
            data, feature_cols,
            n_bootstrap=n_bootstrap,
            estimator_type=estimator_type,
        )
        print_evaluation(eval_results, estimator_type.upper())
        return {'bootstrap_est': bootstrap_est, 'eval': eval_results}
    
    s1 = cache.get_or_run("stage1", _run_s1, s1_params, force_rerun=force_s1)
    bootstrap_est = s1['bootstrap_est']
    print(f"  Stage 1 완료: MAE={s1['eval']['optimal_t_mae']:.4f}, "
          f"위반={s1['eval']['safety_violation_rate']:.1%}")
    
    # ===== Stage 2: Uncertainty =====
    s2_params = {**s1_params, 'alpha': 0.1, 'synthesis': 'max'}
    force_s2 = force or from_stage <= 2
    
    t_grid = np.linspace(0.05, 0.35, 31)
    
    def _run_s2():
        stage2_results = run_stage2(
            data, feature_cols, bootstrap_est, t_grid,
            alpha=0.1, synthesis_method="equal_weight",
        )
        print_stage2_results(stage2_results)
        return stage2_results
    
    s2 = cache.get_or_run("stage2", _run_s2, s2_params, force_rerun=force_s2)
    print(f"  Stage 2 완료: u(x) 평균={s2['diagnostics']['u_combined_mean']:.3f}")
    
    # ===== Stage 3: Deploy =====
    # Stage 3는 가벼우므로 항상 실행
    print("\n" + "-" * 60)
    all_results = run_stage3(
        data, s2,
        baseline_t=0.15,
        lambda_values=[0.0, 0.3, 0.5, 0.7, 1.0],
    )
    print_stage3_results(all_results)
    
    return {
        'data': data,
        'stage1': s1,
        'stage2': s2,
        'stage3': all_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safe Policy Pipeline")
    parser.add_argument('--n-samples', type=int, default=3000)
    parser.add_argument('--endogeneity', type=float, default=0.3)
    parser.add_argument('--heterogeneity', type=float, default=0.5)
    parser.add_argument('--n-bootstrap', type=int, default=20)
    parser.add_argument('--estimator', choices=['gps', 'dml', 'odml'], default='dml')
    parser.add_argument('--from-stage', type=int, default=99,
                        help='이 stage부터 재실행 (0~3)')
    parser.add_argument('--force', action='store_true',
                        help='캐시 무시하고 전체 재실행')
    parser.add_argument('--list-cache', action='store_true',
                        help='캐시 목록만 출력')
    parser.add_argument('--clear-cache', action='store_true',
                        help='캐시 전체 삭제')
    parser.add_argument('--cache-dir', default='./cache')
    
    args = parser.parse_args()
    
    cache = StageCache(args.cache_dir)
    
    if args.list_cache:
        print("저장된 캐시:")
        cache.list_caches()
    elif args.clear_cache:
        cache.clear()
    else:
        run_pipeline(
            n_samples=args.n_samples,
            endogeneity=args.endogeneity,
            heterogeneity=args.heterogeneity,
            n_bootstrap=args.n_bootstrap,
            estimator_type=args.estimator,
            from_stage=args.from_stage,
            force=args.force,
            cache_dir=args.cache_dir,
        )