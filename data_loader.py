"""
M5 데이터 다운로드
=================
사전 준비:
  pip install kaggle
  
  Kaggle API 설정:
  1. https://www.kaggle.com/settings → "Create New Token"
  2. kaggle.json을 ~/.kaggle/kaggle.json에 저장 (Windows: C:\\Users\\<user>\\.kaggle\\)
  3. M5 대회 규칙 동의: https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules

실행:
  python download_m5.py
"""

import zipfile
from pathlib import Path


def download_m5_data(data_dir="./m5_data"):
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    required_files = [
        'sales_train_evaluation.csv',
        'sell_prices.csv',
        'calendar.csv',
    ]

    if all((data_path / f).exists() for f in required_files):
        print("✓ M5 데이터가 이미 존재합니다.")
        return data_path

    print("M5 데이터 다운로드 중...")

    try:
        import kaggle
        kaggle.api.competition_download_files(
            'm5-forecasting-accuracy',
            path=data_dir,
            quiet=False,
        )

        zip_path = data_path / 'm5-forecasting-accuracy.zip'
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(data_path)
            zip_path.unlink()

        print("✓ 다운로드 완료!")
        return data_path

    except Exception as e:
        print(f"\n❌ 다운로드 실패: {e}")
        print("\n수동 다운로드:")
        print("1. https://www.kaggle.com/competitions/m5-forecasting-accuracy/data")
        print("2. 'Download All' → 압축 해제 후 m5_data/ 폴더에 저장")
        return None


if __name__ == "__main__":
    download_m5_data()