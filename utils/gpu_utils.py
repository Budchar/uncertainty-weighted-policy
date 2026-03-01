"""
GPU 가속 모델 팩토리
====================

사용법:
  1. 이 파일을 src/gpu_utils.py로 저장
  2. stage1_estimate.py에서:
     - from sklearn.ensemble import GradientBoostingRegressor 를 삭제
     - from src.gpu_utils import make_regressor 를 추가
     - GradientBoostingRegressor(...) 를 make_regressor(...) 로 교체

이렇게 하면 기존 코드 구조를 거의 안 건드리고 GPU 가속을 적용할 수 있습니다.
"""

import warnings

# XGBoost 사용 가능 여부 확인
_USE_XGBOOST = False
_USE_GPU = False

try:
    import xgboost as xgb
    _USE_XGBOOST = True
    
    # GPU 사용 가능 여부 확인
    try:
        _test = xgb.XGBRegressor(device="cuda", n_estimators=1, verbosity=0)
        import numpy as np
        _test.fit(np.array([[0, 1]]), np.array([0]))
        _USE_GPU = True
        print("[gpu_utils] ✓ XGBoost + CUDA GPU 사용")
    except Exception:
        print("[gpu_utils] ⚠ XGBoost는 있지만 GPU 사용 불가 — CPU 모드로 XGBoost 사용")
except ImportError:
    print("[gpu_utils] ⚠ XGBoost 미설치 — sklearn GradientBoosting 사용 (느림)")


def make_regressor(n_estimators=200, max_depth=5, random_state=42, **kwargs):
    """GradientBoosting 호환 회귀 모델 생성
    
    - XGBoost + GPU 가능: XGBRegressor(device="cuda")
    - XGBoost만 가능: XGBRegressor(CPU)
    - XGBoost 없음: sklearn GradientBoostingRegressor (fallback)
    
    반환된 모델은 .fit(X, y), .predict(X) 인터페이스가 동일합니다.
    """
    
    if _USE_XGBOOST:
        params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            tree_method="hist",
            verbosity=0,
            **kwargs,
        )
        if _USE_GPU:
            params["device"] = "cuda"
        
        return xgb.XGBRegressor(**params)
    
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )