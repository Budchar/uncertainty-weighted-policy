"""
캐시 유틸리티
=============

각 Stage의 결과를 pickle로 저장/로드하여
반복 실행을 방지한다.

사용법:
    from src.cache_utils import StageCache
    
    cache = StageCache("./cache")
    
    # 저장
    cache.save("stage0", {"data": data, "dgp": dgp}, 
               params={"n_samples": 3000, "endogeneity": 0.3})
    
    # 로드 (파라미터가 같으면 캐시 사용)
    result = cache.load("stage0", 
                        params={"n_samples": 3000, "endogeneity": 0.3})
    
    # 또는 간편하게
    result = cache.get_or_run("stage0", run_fn, 
                              params={"n_samples": 3000, "endogeneity": 0.3})
"""

import pickle
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class StageCache:
    """Stage별 결과 캐시 관리"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _params_hash(self, params: Dict) -> str:
        """파라미터를 해시로 변환 (캐시 키)"""
        # 정렬된 JSON → MD5 해시
        param_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(param_str.encode()).hexdigest()[:12]
    
    def _cache_path(self, stage_name: str, params: Dict) -> Path:
        """캐시 파일 경로"""
        h = self._params_hash(params)
        return self.cache_dir / f"{stage_name}_{h}.pkl"
    
    def _meta_path(self, stage_name: str, params: Dict) -> Path:
        """메타 정보 파일 경로 (파라미터 기록용)"""
        h = self._params_hash(params)
        return self.cache_dir / f"{stage_name}_{h}_meta.json"
    
    def exists(self, stage_name: str, params: Dict) -> bool:
        """캐시 존재 여부"""
        return self._cache_path(stage_name, params).exists()
    
    def save(self, stage_name: str, result: Any, params: Dict):
        """결과 저장"""
        cache_path = self._cache_path(stage_name, params)
        meta_path = self._meta_path(stage_name, params)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        
        # 메타 정보도 저장 (디버깅용)
        import datetime
        meta = {
            'stage': stage_name,
            'params': {k: str(v) if not isinstance(v, (int, float, bool, str, list)) else v 
                       for k, v in params.items()},
            'saved_at': datetime.datetime.now().isoformat(),
            'cache_file': cache_path.name,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        print(f"  💾 캐시 저장: {cache_path.name}")
    
    def load(self, stage_name: str, params: Dict) -> Optional[Any]:
        """결과 로드 (없으면 None)"""
        cache_path = self._cache_path(stage_name, params)
        
        if not cache_path.exists():
            return None
        
        with open(cache_path, 'rb') as f:
            result = pickle.load(f)
        
        print(f"  📂 캐시 로드: {cache_path.name}")
        return result
    
    def get_or_run(
        self,
        stage_name: str,
        run_fn: Callable,
        params: Dict,
        force_rerun: bool = False,
    ) -> Any:
        """캐시가 있으면 로드, 없으면 실행 후 저장
        
        Args:
            stage_name: "stage0", "stage1", etc.
            run_fn: 실행 함수 (파라미터 없이 호출 가능한 callable)
            params: 캐시 키가 될 파라미터
            force_rerun: True면 캐시 무시하고 재실행
        """
        if not force_rerun:
            cached = self.load(stage_name, params)
            if cached is not None:
                return cached
        
        # 실행
        result = run_fn()
        
        # 저장
        self.save(stage_name, result, params)
        
        return result
    
    def clear(self, stage_name: Optional[str] = None):
        """캐시 삭제
        
        Args:
            stage_name: 특정 stage만 삭제 (None이면 전체)
        """
        pattern = f"{stage_name}_*" if stage_name else "*"
        deleted = 0
        for f in self.cache_dir.glob(pattern):
            f.unlink()
            deleted += 1
        print(f"  🗑️  캐시 {deleted}개 삭제")
    
    def list_caches(self):
        """저장된 캐시 목록"""
        meta_files = sorted(self.cache_dir.glob("*_meta.json"))
        
        if not meta_files:
            print("  (캐시 없음)")
            return
        
        for mf in meta_files:
            with open(mf) as f:
                meta = json.load(f)
            pkl_path = self.cache_dir / meta['cache_file']
            size_mb = pkl_path.stat().st_size / 1024 / 1024 if pkl_path.exists() else 0
            print(f"  {meta['stage']:>10} | {size_mb:>6.1f}MB | {meta['saved_at'][:19]} | {meta.get('params', {})}")