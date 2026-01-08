#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PET/CT Radiomics Pipeline - Full Analysis Script
=================================================
Executes complete workflow:
1. DICOM to NIfTI conversion
2. PET-to-CT spatial alignment
3. CT segmentation (TotalSegmentator, 104 structures)
4. SUV conversion
5. Radiomic feature extraction
6. Quality control visualization

Usage:
    python run_full_analysis.py [--input DICOM_FOLDER] [--skip-visualization]

=================================================
Note: This script is called by run_analysis.bat (Windows).
      Works on all platforms (Mac/Linux/Windows).
=================================================
"""

import sys
import os
import argparse
from pathlib import Path
import subprocess
import time
from datetime import datetime

# プロジェクトのベースディレクトリ
BASE_DIR = Path(__file__).parent

# Python実行パス（環境に応じて自動検出）
def get_python_path():
    """環境に応じたPythonパスを取得"""
    # 現在実行中のPythonを使用（conda環境対応）
    return sys.executable

PYTHON_PATH = get_python_path()

# 色付き出力用
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_step(step_num, text):
    print(f"{Colors.BLUE}{Colors.BOLD}[Step {step_num}]{Colors.END} {text}")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def run_script(script_name, description):
    """スクリプトを実行"""
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        print_error(f"Script not found: {script_name}")
        return False

    print(f"  Running {script_name}...")
    try:
        result = subprocess.run(
            [PYTHON_PATH, str(script_path)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=600  # 10分タイムアウト
        )
        if result.returncode == 0:
            print_success(description)
            return True
        else:
            print_error(f"Error in {script_name}")
            if result.stderr:
                print(f"  {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print_error(f"Timeout: {script_name}")
        return False
    except Exception as e:
        print_error(f"Exception: {e}")
        return False

def check_new_data():
    """新しいDICOMデータがあるか確認（フォルダ名=患者ID方式）"""
    dicom_dir = BASE_DIR / "raw_download"
    seg_dir = BASE_DIR / "segmentations"

    if not dicom_dir.exists():
        return [], []

    # 処理済み患者（segmentationsフォルダに存在するもの）
    processed = set()
    if seg_dir.exists():
        processed = {p.name for p in seg_dir.iterdir() if p.is_dir()}

    # 新しいフォルダを検出
    all_folders = []
    new_folders = []
    for folder in dicom_dir.iterdir():
        if folder.is_dir():
            all_folders.append(folder.name)
            if folder.name not in processed:
                new_folders.append(folder.name)

    return new_folders, all_folders

def main():
    parser = argparse.ArgumentParser(
        description='PET/CT Radiomics 統合解析パイプライン',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--skip-visualization', action='store_true',
                        help='可視化をスキップする')
    parser.add_argument('--force', action='store_true',
                        help='既存データも再処理する')
    parser.add_argument('--visualize-only', action='store_true',
                        help='可視化のみ実行する')
    args = parser.parse_args()

    start_time = time.time()

    print_header("PET/CT Radiomics 統合解析パイプライン")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"作業ディレクトリ: {BASE_DIR}")
    print(f"Python: {PYTHON_PATH}")

    # 可視化のみモード
    if args.visualize_only:
        print_header("可視化のみ実行")

        print_step(1, "肺セグメンテーション確認画像")
        run_script("visualize_lung_segmentation.py", "肺セグメンテーション画像を生成")

        print_step(2, "PET-CT位置合わせ確認画像")
        run_script("visualize_ct_pet_seg.py", "PET-CT位置合わせ画像を生成")

        print_step(3, "SUV Radiomics結果グラフ")
        run_script("plot_suv_results.py", "SUV結果グラフを生成")

        print_header("完了")
        print(f"結果は analysis_results/ フォルダに保存されました")
        return

    # 新しいデータの確認
    new_data, all_data = check_new_data()
    print(f"\n患者フォルダ: {len(all_data)}件")
    for folder in all_data:
        status = "NEW" if folder in new_data else "processed"
        print(f"  - {folder} [{status}]")

    if new_data:
        print(f"\n新規データ: {len(new_data)}件")
    elif not args.force:
        print_warning("新しいデータがありません。--force オプションで再処理できます。")

    # Step 1: DICOM変換・セグメンテーション・Radiomics
    print_step(1, "DICOM変換・セグメンテーション・Radiomics抽出")
    if not run_script("run_pipeline.py", "基本パイプライン完了"):
        print_error("基本パイプラインでエラーが発生しました")
        if not args.force:
            return

    # Step 2: SUV補正とRadiomics再計算
    print_step(2, "SUV補正とRadiomics再計算")
    if not run_script("create_final_suv.py", "SUV補正完了"):
        print_warning("SUV補正でエラーが発生しました（続行）")

    # Step 3: 可視化
    if not args.skip_visualization:
        print_step(3, "可視化・レポート生成")

        print("  3.1 肺セグメンテーション確認画像...")
        run_script("visualize_lung_segmentation.py", "肺セグメンテーション画像")

        print("  3.2 PET-CT位置合わせ確認画像...")
        run_script("visualize_ct_pet_seg.py", "PET-CT位置合わせ画像")

        print("  3.3 SUV Radiomics結果グラフ...")
        run_script("plot_suv_results.py", "SUV結果グラフ")

    # 完了
    elapsed = time.time() - start_time
    print_header("解析完了")
    print(f"処理時間: {elapsed/60:.1f} 分")
    print(f"\n出力ファイル:")
    print(f"  - pet_ct_radiomics_results.csv  (Radiomics特徴量)")
    print(f"  - analysis_results/             (可視化画像)")
    print(f"  - nifti_images/                 (NIfTI画像)")
    print(f"  - segmentations/                (セグメンテーション)")

if __name__ == "__main__":
    main()
