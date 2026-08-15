"""직교화 DML 진단 — 직교화가 이 DGP에서 편향을 줄이는가

배경:
    DMLDoseResponseEstimator는 cross-fitting 잔차를 계산만 하고 최종 outcome
    모델을 원본 (X, T) → Y 로 학습하므로 직교화가 추정에 반영되지 않는다.
    OrthogonalDMLEstimator는 처리 변수를 잔차화된 기저로만 투입해 직교화를
    실제로 적용한다. 두 추정기의 차이가 결과에 어떤 영향을 주는지 확인한다.

핵심 논점:
    이 DGP의 내생성은 **미관측** 교란 U에서 온다 (U는 get_feature_columns()에
    포함되지 않는다). DML의 직교화는 관측된 X를 통한 교란 경로만 제거하므로,
    U가 관측되지 않으면 이론적으로 편향을 줄일 수 없다.

    이를 판별하기 위해 두 조건을 비교한다:
      조건 A (U 미관측, 기본 설정)  — 직교화가 편향을 못 줄여야 이론과 일치
      조건 B (U 관측, 오라클 공변량) — 두 추정기 모두 편향이 사라져야 정상

    조건 B에서 편향이 사라지면 OrthogonalDMLEstimator 구현이 정상임을
    확인하는 동시에, 조건 A의 결과가 구현 결함이 아니라 식별(identification)
    한계임을 입증한다.

실행:
    python run_odml_diagnostic.py
    python run_odml_diagnostic.py --n-samples 5000 --repeats 5
"""
import argparse
import sys
import numpy as np

from src.stage0_dgp import create_dataset
from src.stage1_estimate import DMLDoseResponseEstimator, OrthogonalDMLEstimator

T_GRID = np.linspace(0.05, 0.35, 31)


def run_one(data, dgp, use_u: bool, estimator: str):
    """단일 적합 → 추정 최적 마진의 오차 지표 반환"""
    cols = [c for c in dgp.get_feature_columns() if c in data.columns]
    X = data[cols].values
    if use_u:
        # 오라클 조건: 미관측 교란을 공변량으로 노출시킨다
        X = np.column_stack([X, data['unobserved_confounder'].values])

    T = data['treatment'].values
    Y = data['outcome'].values
    true_opt = data['true_optimal_margin'].values

    if estimator == 'dml':
        est = DMLDoseResponseEstimator(n_folds=5, n_estimators=100, max_depth=4)
    else:
        est = OrthogonalDMLEstimator(
            n_folds=5, n_estimators=100, max_depth=4, basis_degree=2
        )
    est.fit(X, T, Y)

    opt = T_GRID[np.argmax(est.predict(X, T_GRID), axis=1)]
    return {
        'mae': float(np.mean(np.abs(opt - true_opt))),
        'bias': float(np.mean(opt - true_opt)),
        'std': float(np.std(opt)),
        'true_std': float(np.std(true_opt)),
    }


def main():
    parser = argparse.ArgumentParser(description="직교화 DML 진단")
    parser.add_argument('--n-samples', type=int, default=3000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--heterogeneity', type=float, default=0.5)
    parser.add_argument('--endogeneity', type=float, nargs='+',
                        default=[0.0, 0.1, 0.3, 0.5, 0.8])
    args = parser.parse_args()

    print(f"\n{'조건':<20}{'내생성':>7}{'추정기':>7}"
          f"{'MAE':>10}{'bias':>10}{'추정std':>10}{'참std':>9}")
    print("-" * 74)

    for endo in args.endogeneity:
        for cond_label, use_u in [("A: U 미관측", False), ("B: U 관측(오라클)", True)]:
            acc = {}
            for r in range(args.repeats):
                data, dgp = create_dataset(
                    n_samples=args.n_samples,
                    endogeneity=endo,
                    heterogeneity=args.heterogeneity,
                    random_seed=42 + 1000 * r,
                )
                for name in ['dml', 'odml']:
                    m = run_one(data, dgp, use_u, name)
                    a = acc.setdefault(name, {k: [] for k in m})
                    for k, v in m.items():
                        a[k].append(v)

            for name in ['dml', 'odml']:
                a = acc[name]
                print(f"{cond_label:<20}{endo:>7.1f}{name:>7}"
                      f"{np.mean(a['mae']):>10.5f}{np.mean(a['bias']):>+10.5f}"
                      f"{np.mean(a['std']):>10.5f}{np.mean(a['true_std']):>9.5f}")
        print("-" * 74)

    print("\n해석 가이드:")
    print("  - 조건 A에서 두 추정기 모두 bias가 크게 남으면 → 미관측 교란에 의한")
    print("    식별 한계이며 추정기 교체로 해결되지 않음 (안전 레이어의 근거)")
    print("  - 조건 B에서 bias가 0 근처로 떨어지면 → odml 구현이 정상이며,")
    print("    직교화 자체는 관측된 교란에 대해 제대로 작동함")


if __name__ == "__main__":
    sys.exit(main())
