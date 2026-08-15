"""논문 표1 형식으로 dml(원본) vs odml(직교화) 비교

실행: python run_odml_comparison.py

논문 §3.2의 대표 파라미터 선정 규칙을 그대로 구현한다:
  "36개 시나리오 전체에서 Naive 수익 이상을 유지하면서 위반율을 최소화하는 값"
  Threshold는 모든 τ가 조건을 만족하지 못하므로 "조건을 가장 많이 만족한" τ를 택한다.

먼저 dml 원본 데이터로 규칙을 검증한다(λ=0.5, c=0.20, τ=0.85가 재현되어야 함).
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

SCEN = ["endogeneity", "n_samples", "heterogeneity"]
RUN = SCEN + ["repeat"]


def load(path):
    d = pd.read_csv(path)
    assert d.groupby(RUN).ngroups == 360, f"360런 아님: {path}"
    return d


def select_param(d, prefix, exclude=()):
    """논문 규칙으로 대표 파라미터 선정 → (이름, 만족 시나리오 수, 전체 만족 여부)"""
    cands = [p for p in d.policy.unique()
             if p.startswith(prefix) and p not in exclude]
    # 시나리오별 평균
    sc = d.groupby(["policy"] + SCEN)["improvement_over_baseline"].mean()
    naive = sc.loc["naive_optimal"]
    overall = d.groupby("policy")[["safety_violation_rate"]].mean()

    rows = []
    for p in cands:
        ok = int((sc.loc[p] >= naive - 1e-12).sum())
        rows.append((p, ok, float(overall.loc[p, "safety_violation_rate"])))
    n_scen = len(naive)
    full = [r for r in rows if r[1] == n_scen]
    if full:
        best = min(full, key=lambda r: r[2])
        return best[0], best[1], True
    best = max(rows, key=lambda r: (r[1], -r[2]))
    return best[0], best[1], False


def table(d, label, picks):
    g = d.groupby("policy")[
        ["safety_violation_rate", "improvement_over_baseline", "mean_regret"]
    ].mean()
    print(f"\n  [{label}]")
    print(f"  {'정책':<22}{'위반율':>9}{'수익개선':>10}{'Regret':>10}")
    print("  " + "-" * 51)
    for name, pol in picks:
        r = g.loc[pol]
        print(f"  {name:<22}{r.safety_violation_rate*100:>8.2f}%"
              f"{r.improvement_over_baseline:>9.2f}%{r.mean_regret:>10.5f}")
    return g


def paired(d, a, b, label):
    out = []
    for col, mult in [("safety_violation_rate", 100), ("improvement_over_baseline", 1)]:
        x = d[d.policy == a].set_index(RUN)[col].sort_index()
        y = d[d.policy == b].set_index(RUN)[col].sort_index()
        assert x.index.equals(y.index)
        t, p = stats.ttest_rel(x, y)
        out.append(((x - y).mean() * mult, t, p))
    print(f"  {label:<28}Δ위반 {out[0][0]:+7.2f}%p (p={out[0][2]:.1e})   "
          f"Δ수익 {out[1][0]:+6.2f}%p (p={out[1][2]:.1e})")


print("=" * 72)
print("1. 선정 규칙 검증 — dml 원본에서 논문의 λ=0.5 / c=0.20 / τ=0.85 재현")
print("=" * 72)
dml = load("results/summary_20260416_200920.csv")
for pref, exc in [("uw_lambda", ("uw_lambda0.0",)), ("uniform_c", ()), ("threshold_", ())]:
    p, ok, full = select_param(dml, pref, exc)
    print(f"  {pref:<12} → {p:<18} (36개 중 {ok}개 만족, 전체만족={full})")

print()
print("=" * 72)
print("2. 논문 표1 형식 비교")
print("=" * 72)

odml = load("results/summary_odml_paper_20260813_231553.csv")

for label, d in [("dml (논문 원본, 비직교)", dml), ("odml (직교화 적용)", odml)]:
    picks = [("Oracle", "oracle")]
    for pref, exc, nm in [("uw_lambda", ("uw_lambda0.0",), "UW"),
                          ("uniform_c", (), "Uniform"),
                          ("threshold_", (), "Threshold")]:
        p, ok, full = select_param(d, pref, exc)
        tag = "" if full else f" [{ok}/36]"
        picks.append((f"{nm} ({p.split('_')[-1]}){tag}", p))
    picks += [("Naive", "naive_optimal"), ("Baseline", "baseline")]
    table(d, label, picks)

print()
print("=" * 72)
print("3. 안전 레이어 기여도 — 추정기별 (paired t-test, n=360)")
print("=" * 72)
for label, d in [("dml ", dml), ("odml", odml)]:
    uw, _, _ = select_param(d, "uw_lambda", ("uw_lambda0.0",))
    uni, _, _ = select_param(d, "uniform_c", ())
    th, _, _ = select_param(d, "threshold_", ())
    print(f"\n  [{label}]  대표 UW = {uw}")
    paired(d, uw, "naive_optimal", "UW vs Naive")
    paired(d, uw, uni, f"UW vs {uni}")
    paired(d, uw, th, f"UW vs {th}")

print()
print("=" * 72)
print("4. 추정 정확도 (Stage 1) — 잔차 사용이 추정 자체에 미친 영향")
print("=" * 72)
for label, d in [("dml ", dml), ("odml", odml)]:
    if "s1_optimal_t_mae" in d.columns:
        s = d.groupby(RUN)["s1_optimal_t_mae"].first()
        print(f"  {label}: optimal_t MAE 평균 {s.mean():.5f}  (표준편차 {s.std():.5f})")
