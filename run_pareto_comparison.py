"""UW vs Uniform — 파레토 프론티어 기준 비교 (정정판)

앞선 분석의 결함:
    λ에 대해 수익이 단조가 아니다(작은 λ에서 오히려 수익이 낮고 위반은 높다).
    점들을 수익 기준으로 정렬한 뒤 np.interp를 쓰면, 실제로는 지배당하는
    점들이 곡선에 섞여 들어가 존재하지 않는 구간을 만든다.

정정:
    각 방법군에서 파레토 효율 점만 남긴다(자기보다 수익이 높으면서 위반이
    낮은 점이 없는 것). 그 위에서만 비교한다.
"""
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
rng = np.random.default_rng(42)
RUN = ["endogeneity", "n_samples", "heterogeneity", "repeat"]
B = 500


def prep(path):
    d = pd.read_csv(path, usecols=RUN + ["policy", "safety_violation_rate",
                                         "improvement_over_baseline"])
    d = d[d.policy.str.startswith(("uw_lambda", "uniform_c")) |
          d.policy.isin(["naive_optimal"])]
    vi = d.pivot_table(index=RUN, columns="policy", values="safety_violation_rate").sort_index()
    re = d.pivot_table(index=RUN, columns="policy", values="improvement_over_baseline").sort_index()
    return vi, re


def pareto(rev, vio):
    """수익 최대·위반 최소 기준 파레토 효율 점의 인덱스 (수익 오름차순 반환)"""
    keep = []
    for i in range(len(rev)):
        dominated = np.any((rev >= rev[i]) & (vio <= vio[i]) &
                           ((rev > rev[i]) | (vio < vio[i])))
        if not dominated:
            keep.append(i)
    keep = np.array(keep)
    o = np.argsort(rev[keep])
    return keep[o]


def front(vi, re, pols, idx):
    r = re.iloc[idx][pols].mean().values
    v = vi.iloc[idx][pols].mean().values
    k = pareto(r, v)
    return r[k], v[k]


def at(x, xs, ys):
    if len(xs) < 2 or x < xs.min() or x > xs.max():
        return np.nan
    return float(np.interp(x, xs, ys))


for label, path in [("dml (논문 원본)", "results/summary_20260416_200920.csv"),
                    ("odml (직교화)", "results/summary_odml_paper_20260813_231553.csv")]:
    vi, re = prep(path)
    UW = [c for c in vi.columns if c.startswith("uw_lambda") and c != "uw_lambda0.0"]
    UN = [c for c in vi.columns if c.startswith("uniform_c")]
    n = len(vi)
    full = np.arange(n)

    ru, vu = front(vi, re, UW, full)
    rn, vn = front(vi, re, UN, full)

    print("=" * 76)
    print(f"{label}   (Naive 수익 {re['naive_optimal'].mean():+.2f}%)")
    print("=" * 76)
    print(f"  파레토 효율 점 수 — UW {len(ru)}/{len(UW)},  Uniform {len(rn)}/{len(UN)}")
    print(f"  UW      수익 범위 [{ru.min():+.2f}, {ru.max():+.2f}]%")
    print(f"  Uniform 수익 범위 [{rn.min():+.2f}, {rn.max():+.2f}]%")

    lo, hi = max(ru.min(), rn.min()), min(ru.max(), rn.max())
    if hi <= lo:
        print("  겹치는 수익 구간 없음")
        continue
    targets = np.linspace(lo, hi, 6)[1:-1]

    print(f"\n  같은 수익 수준에서의 위반율  (UW − Uniform, 음수면 UW 우세)")
    print(f"  {'수익':>8}{'UW':>9}{'Uniform':>10}{'차이':>9}{'95% CI (bootstrap)':>24}")
    print("  " + "-" * 60)
    for t in targets:
        boot = []
        for _ in range(B):
            idx = rng.integers(0, n, n)
            a = at(t, *front(vi, re, UW, idx))
            c = at(t, *front(vi, re, UN, idx))
            if not (np.isnan(a) or np.isnan(c)):
                boot.append(a - c)
        boot = np.array(boot) * 100
        d0 = (at(t, ru, vu) - at(t, rn, vn)) * 100
        if len(boot) < B * 0.5:
            print(f"  {t:>7.2f}%  (재표집에서 구간 이탈 잦음 — 판정 보류)")
            continue
        ci = np.percentile(boot, [2.5, 97.5])
        sig = "유의" if ci[0] * ci[1] > 0 else "n.s."
        print(f"  {t:>7.2f}%{at(t,ru,vu)*100:>8.2f}%{at(t,rn,vn)*100:>9.2f}%"
              f"{d0:>+8.2f}p   [{ci[0]:+6.2f}, {ci[1]:+6.2f}] {sig}")
    print()
