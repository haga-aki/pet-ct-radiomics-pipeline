"""
PET-CT Radiomics Pipeline
=========================

DICOMからNIfTI変換 → TotalSegmentatorでセグメンテーション → PyRadiomicsで特徴量抽出

改善点 (v2.0):
- DICOMシリーズ自動選別機能（本体CTとPETを自動判別）
- 依存関係バージョン固定（PyTorch 2.4+, NumPy 1.x）
- 入力検証の強化
"""

import sys
import os
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import dicom2nifti
from totalsegmentator.python_api import totalsegmentator
from radiomics import featureextractor
import pandas as pd
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import yaml

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not available. DICOM series auto-detection disabled.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. GPU features disabled.")


def load_config(config_path="config.yaml"):
    """設定ファイルの読み込み"""
    default_config = {
        'organs': [
            'liver',
            'kidney_left', 'kidney_right',
            'adrenal_gland_left', 'adrenal_gland_right',
            'vertebrae_L1',
            'aorta',
            'lung_upper_lobe_left', 'lung_lower_lobe_left'  # 左肺
        ],
        'modalities': ['CT', 'PET'],
        'segmentation': {
            'tasks': {'CT': 'total', 'MR': 'total_mr', 'PET': 'use_ct_mask', 'SPECT': 'use_ct_mask'},
            'fast': True
        },
        'output': {'csv_file': 'radiomics_results.csv', 'include_diagnostics': False},
        'dicom_selection': {
            'auto_select': True,  # シリーズ自動選別
            'min_ct_slices': 100,  # CTの最小スライス数
            'min_pet_slices': 50,  # PETの最小スライス数
        }
    }
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            if user_config:
                for key in user_config:
                    if isinstance(user_config[key], dict):
                        default_config.setdefault(key, {}).update(user_config[key])
                    else:
                        default_config[key] = user_config[key]
    return default_config


def get_root_dir():
    """ルートディレクトリの取得"""
    if os.environ.get("PET_PIPELINE_ROOT"):
        return Path(os.environ["PET_PIPELINE_ROOT"])
    return Path(__file__).parent.resolve()


ROOT_DIR = get_root_dir()
DICOM_DIR = ROOT_DIR / "raw_download"
NIFTI_DIR = ROOT_DIR / "nifti_images"
SEG_DIR = ROOT_DIR / "segmentations"
CONFIG = load_config()
RESULT_CSV = CONFIG['output']['csv_file']
ID_MAP_FILE = Path("id_mapping.csv")

for p in [NIFTI_DIR, SEG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def check_gpu_memory():
    """GPU可用性とメモリ状態の確認"""
    if not TORCH_AVAILABLE:
        return False, "PyTorch not available"

    if not torch.cuda.is_available():
        return False, "CUDA not available"

    try:
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        gpu_mem_free = gpu_mem_total - gpu_mem_allocated

        print(f"\n=== GPU Status ===")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {gpu_mem_total:.2f} GB")
        print(f"Free: {gpu_mem_free:.2f} GB")

        if gpu_mem_free < 2.0:
            return False, f"Insufficient GPU memory (only {gpu_mem_free:.2f} GB free)"

        return True, f"GPU ready ({gpu_mem_free:.2f} GB available)"
    except Exception as e:
        return False, f"GPU check failed: {e}"


def log_progress(message, level="INFO"):
    """進捗メッセージの出力"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def get_or_create_anon_id(original_folder_name):
    """匿名IDの取得または作成"""
    mapping = {}
    if ID_MAP_FILE.exists():
        with open(ID_MAP_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    mapping[row[0]] = row[1]
    if original_folder_name in mapping:
        return mapping[original_folder_name]
    current_count = len(mapping)
    new_id = f"ILD_{current_count + 1:03d}"
    is_new_file = not ID_MAP_FILE.exists()
    with open(ID_MAP_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["Original_Folder_Name", "Anonymized_ID"])
        writer.writerow([original_folder_name, new_id])
    print(f"  [New ID] {original_folder_name} -> {new_id}")
    return new_id


def scan_dicom_series(patient_path):
    """
    DICOMシリーズをスキャンして情報を収集

    Returns:
        list of dict: 各シリーズの情報
            - path: シリーズのパス
            - modality: CT/PT/MR等
            - series_description: シリーズ説明
            - num_slices: スライス数
            - image_size: 画像サイズ (rows, cols)
    """
    if not PYDICOM_AVAILABLE:
        return []

    series_info = []

    # サブフォルダを探索
    subdirs = [d for d in patient_path.iterdir() if d.is_dir()]
    if not subdirs:
        subdirs = [patient_path]

    for subdir in subdirs:
        dcm_files = list(subdir.glob("*.dcm")) + list(subdir.glob("*.DCM"))
        if not dcm_files:
            # ファイル拡張子なしのDICOMも検索
            dcm_files = [f for f in subdir.iterdir() if f.is_file() and not f.suffix]

        if not dcm_files:
            continue

        try:
            # 最初のDICOMファイルからメタデータを取得
            dcm = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)

            modality = getattr(dcm, 'Modality', 'UNKNOWN')
            series_desc = getattr(dcm, 'SeriesDescription', '')
            rows = getattr(dcm, 'Rows', 0)
            cols = getattr(dcm, 'Columns', 0)

            series_info.append({
                'path': subdir,
                'modality': modality,
                'series_description': series_desc,
                'num_slices': len(dcm_files),
                'image_size': (rows, cols),
                'folder_name': subdir.name
            })
        except Exception as e:
            log_progress(f"  Warning: Could not read DICOM in {subdir.name}: {e}", "WARNING")

    return series_info


def select_best_series(series_info, config):
    """
    CT/PETの本体シリーズを自動選別

    選別基準:
    - CT: スライス数が最大で、かつ画像サイズが512x512のシリーズ
    - PET: スライス数が最大で、かつ"Axial"を含むか画像サイズが妥当なシリーズ

    Returns:
        dict: {'CT': path, 'PET': path} またはNone
    """
    selection_config = config.get('dicom_selection', {})
    min_ct_slices = selection_config.get('min_ct_slices', 100)
    min_pet_slices = selection_config.get('min_pet_slices', 50)

    selected = {}

    # CTシリーズの選別
    ct_series = [s for s in series_info if s['modality'] == 'CT']
    if ct_series:
        # スライス数でフィルタリング（レポートやDose Report除外）
        valid_ct = [s for s in ct_series if s['num_slices'] >= min_ct_slices]

        if valid_ct:
            # 512x512の標準CTを優先
            standard_ct = [s for s in valid_ct if s['image_size'] == (512, 512)]
            if standard_ct:
                # スライス数が最も多いものを選択
                best_ct = max(standard_ct, key=lambda x: x['num_slices'])
            else:
                best_ct = max(valid_ct, key=lambda x: x['num_slices'])

            selected['CT'] = best_ct['path']
            log_progress(f"  Selected CT: {best_ct['folder_name']} "
                        f"({best_ct['num_slices']} slices, {best_ct['image_size']}, "
                        f"'{best_ct['series_description']}')", "INFO")

    # PETシリーズの選別
    pet_series = [s for s in series_info if s['modality'] in ['PT', 'PET']]
    if pet_series:
        # スライス数でフィルタリング
        valid_pet = [s for s in pet_series if s['num_slices'] >= min_pet_slices]

        if valid_pet:
            # "Axial"を含むシリーズを優先
            axial_pet = [s for s in valid_pet
                        if 'axial' in s['series_description'].lower()
                        or 'body' in s['series_description'].lower()]

            if axial_pet:
                best_pet = max(axial_pet, key=lambda x: x['num_slices'])
            else:
                best_pet = max(valid_pet, key=lambda x: x['num_slices'])

            selected['PET'] = best_pet['path']
            log_progress(f"  Selected PET: {best_pet['folder_name']} "
                        f"({best_pet['num_slices']} slices, {best_pet['image_size']}, "
                        f"'{best_pet['series_description']}')", "INFO")

    return selected if selected else None


def detect_folder_structure(patient_path, config):
    """
    フォルダ構造を検出し、適切なモダリティパスを返す

    改善: DICOMシリーズ自動選別機能を追加
    """
    modalities = config['modalities']
    found = {}

    # 従来の方式: CT/PETサブフォルダがある場合
    for mod in modalities:
        mod_path = patient_path / mod
        if mod_path.exists() and mod_path.is_dir():
            found[mod] = mod_path

    if found:
        return found

    # 新方式: DICOMシリーズ自動選別
    if config.get('dicom_selection', {}).get('auto_select', True) and PYDICOM_AVAILABLE:
        log_progress(f"  Scanning DICOM series in {patient_path.name}...", "INFO")
        series_info = scan_dicom_series(patient_path)

        if series_info:
            log_progress(f"  Found {len(series_info)} series:", "INFO")
            for s in series_info:
                log_progress(f"    - {s['folder_name']}: {s['modality']}, "
                            f"{s['num_slices']} slices, {s['image_size']}, "
                            f"'{s['series_description']}'", "INFO")

            selected = select_best_series(series_info, config)
            if selected:
                return selected

    # フォールバック: 直接DICOM処理
    found['CT'] = patient_path
    return found


def step1_convert_dicom(dicom_path, anon_id, modality):
    """DICOM → NIfTI変換"""
    output_path = NIFTI_DIR / f"{anon_id}_{modality}.nii.gz"
    if output_path.exists():
        log_progress(f"{modality} NIfTI exists. Skipping.", "INFO")
        return output_path
    if not dicom_path.exists():
        log_progress(f"DICOM not found: {dicom_path}", "WARNING")
        return None
    try:
        log_progress(f"Converting {modality} DICOM to NIfTI...", "INFO")
        dicom2nifti.dicom_series_to_nifti(dicom_path, output_path)
        log_progress(f"{modality} Conversion completed.", "INFO")
        return output_path
    except Exception as e:
        log_progress(f"{modality} conversion error: {e}", "ERROR")
        return None


def step2_segmentation(nifti_path, anon_id, modality, ct_seg_dir=None, use_gpu=True):
    """TotalSegmentatorによるセグメンテーション"""
    seg_config = CONFIG['segmentation']
    tasks = seg_config.get('tasks', {})
    task = tasks.get(modality, 'total')

    if task == 'use_ct_mask':
        if ct_seg_dir and ct_seg_dir.exists():
            log_progress(f"{modality}: Using CT masks", "INFO")
            return ct_seg_dir
        else:
            log_progress(f"{modality}: No CT segmentation, skipping", "WARNING")
            return None

    output_folder = SEG_DIR / f"{anon_id}_{modality}"
    output_folder.mkdir(parents=True, exist_ok=True)
    combined_path = output_folder / "combined_all_organs.nii.gz"

    if combined_path.exists():
        log_progress(f"{modality}: Segmentation already completed. Skipping.", "INFO")
        return output_folder

    try:
        sample = output_folder / "liver.nii.gz"
        if not sample.exists():
            roi_subset = seg_config.get('roi_subset', None)
            device = 'gpu' if use_gpu else 'cpu'

            if use_gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'

            if roi_subset:
                log_progress(f"{modality}: Running TotalSegmentator (task={task}, "
                            f"roi_subset={len(roi_subset)} organs, device={device})...", "INFO")
                totalsegmentator(input=nifti_path, output=output_folder, task=task,
                               fast=seg_config.get('fast', True), device=device, roi_subset=roi_subset)
            else:
                log_progress(f"{modality}: Running TotalSegmentator (task={task}, device={device})...", "INFO")
                totalsegmentator(input=nifti_path, output=output_folder, task=task,
                               fast=seg_config.get('fast', True), device=device)

        # 結合マスクの作成
        all_files = sorted(list(output_folder.glob("*.nii.gz")))
        input_files = [p for p in all_files if "combined_" not in p.name]
        if not input_files:
            return output_folder

        base_img = nib.load(str(input_files[0]))
        combined_data = np.zeros(base_img.get_fdata().shape, dtype=np.uint16)
        for idx, part_path in enumerate(input_files, start=1):
            try:
                part_data = nib.load(str(part_path)).get_fdata()
                combined_data[part_data > 0] = idx
            except:
                pass

        new_img = nib.Nifti1Image(combined_data, base_img.affine, base_img.header)
        nib.save(new_img, combined_path)
        log_progress(f"{modality}: Segmentation completed successfully.", "INFO")
        return output_folder
    except Exception as e:
        log_progress(f"{modality} segmentation error: {e}", "ERROR")
        return None


def resample_mask_to_image(mask_path, ref_image_path, output_path):
    """マスクを参照画像空間にリサンプリング"""
    try:
        mask_img = sitk.ReadImage(str(mask_path))
        ref_img = sitk.ReadImage(str(ref_image_path))

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)

        resampled_mask = resampler.Execute(mask_img)
        sitk.WriteImage(resampled_mask, str(output_path))
        return True
    except Exception as e:
        print(f"    - Resample error: {e}")
        return False


def resample_pet_to_ct_space(pet_path, ct_path, output_path):
    """PET画像をCT空間にリサンプリング"""
    from nibabel.processing import resample_from_to

    try:
        pet_img = nib.load(str(pet_path))
        ct_img = nib.load(str(ct_path))

        log_progress(f"  Resampling PET to CT space using world coordinates...", "INFO")
        log_progress(f"    PET shape: {pet_img.shape}, CT shape: {ct_img.shape}", "INFO")

        pet_resampled = resample_from_to(pet_img, ct_img, order=1)
        nib.save(pet_resampled, str(output_path))

        log_progress(f"    Resampled PET shape: {pet_resampled.shape}", "INFO")
        log_progress(f"  PET resampled to CT space: {output_path}", "INFO")
        return True

    except Exception as e:
        log_progress(f"  PET resampling error: {e}", "ERROR")
        return False


def step3_radiomics(nifti_path, seg_folder, anon_id, modality):
    """PyRadiomicsによる特徴量抽出"""
    organs = CONFIG['organs']
    include_diag = CONFIG['output'].get('include_diagnostics', False)

    if 'all' in organs:
        files = list(seg_folder.glob("*.nii.gz"))
        organs = [p.stem.replace('.nii', '') for p in files if 'combined_' not in p.name]

    needs_resampling = modality in ['PET', 'PT', 'SPECT']

    log_progress(f"{modality}: Extracting radiomics features for {len(organs)} organs...", "INFO")
    extractor = featureextractor.RadiomicsFeatureExtractor()
    features_list = []

    for idx, organ in enumerate(organs, 1):
        mask_path = seg_folder / f"{organ}.nii.gz"
        if not mask_path.exists():
            continue
        try:
            if needs_resampling:
                resampled_dir = seg_folder.parent / f"{anon_id}_{modality}_resampled"
                resampled_dir.mkdir(parents=True, exist_ok=True)
                resampled_mask_path = resampled_dir / f"{organ}.nii.gz"

                if not resampled_mask_path.exists():
                    print(f"    - Resampling {organ} mask to {modality} space...")
                    if not resample_mask_to_image(mask_path, nifti_path, resampled_mask_path):
                        continue

                mask_path_to_use = str(resampled_mask_path)
            else:
                mask_path_to_use = str(mask_path)

            result = extractor.execute(str(nifti_path), mask_path_to_use)
            row = {"PatientID": anon_id, "Modality": modality, "Organ": organ}
            for k, v in result.items():
                if "original_" in k:
                    row[k] = v
                elif include_diag and "diagnostics_" in k:
                    row[k] = v
            features_list.append(row)
            log_progress(f"    [{idx}/{len(organs)}] Extracted: {organ}", "INFO")
        except Exception as e:
            log_progress(f"    [{idx}/{len(organs)}] Error extracting {organ}: {e}", "ERROR")

    log_progress(f"{modality}: Radiomics extraction completed "
                f"({len(features_list)}/{len(organs)} successful).", "INFO")
    return features_list


def print_version_report():
    """依存関係のバージョン情報を表示"""
    import platform
    print("=" * 60)
    print("PET-CT Radiomics Pipeline - Version Report")
    print("=" * 60)
    print(f"\nSystem Information:")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Python: {platform.python_version()}")

    print(f"\nCore Dependencies:")
    try:
        import radiomics
        print(f"  PyRadiomics: {radiomics.__version__}")
    except:
        print("  PyRadiomics: Not installed")

    try:
        print(f"  SimpleITK: {sitk.Version.VersionString()}")
    except:
        print("  SimpleITK: Not installed")

    try:
        import totalsegmentator
        print(f"  TotalSegmentator: {getattr(totalsegmentator, '__version__', 'installed')}")
    except:
        print("  TotalSegmentator: Not installed")

    print(f"  NumPy: {np.__version__}")
    print(f"  Pandas: {pd.__version__}")
    print(f"  nibabel: {nib.__version__}")

    try:
        print(f"  pydicom: {pydicom.__version__}")
    except:
        print("  pydicom: Not installed")

    if TORCH_AVAILABLE:
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  PyTorch: Not installed")

    print(f"\nConfiguration:")
    print(f"  Config file: {'config.yaml' if Path('config.yaml').exists() else 'Using defaults'}")
    print(f"  Target organs: {CONFIG['organs'][:3]}... ({len(CONFIG['organs'])} total)")
    print(f"  Modalities: {CONFIG['modalities']}")
    print("=" * 60)


def run_dry_run():
    """ドライラン（処理なしのプレビュー）"""
    print("=" * 60)
    print("PET-CT Radiomics Pipeline - Dry Run")
    print("=" * 60)

    print(f"\nInput Directory: {DICOM_DIR}")
    if not DICOM_DIR.exists():
        print(f"  WARNING: Directory does not exist!")
        return

    target_folders = [p.name for p in DICOM_DIR.iterdir() if p.is_dir()]
    print(f"  Found {len(target_folders)} patient folders")

    # 各フォルダのDICOMシリーズをスキャン
    if PYDICOM_AVAILABLE:
        print(f"\n--- DICOM Series Analysis ---")
        for folder in target_folders[:3]:  # 最初の3症例のみ
            patient_path = DICOM_DIR / folder
            series_info = scan_dicom_series(patient_path)
            print(f"\n  {folder}:")
            for s in series_info:
                print(f"    - {s['folder_name']}: {s['modality']}, "
                      f"{s['num_slices']} slices, '{s['series_description']}'")

    print(f"\nOutput Configuration:")
    print(f"  NIfTI Directory: {NIFTI_DIR}")
    print(f"  Segmentation Directory: {SEG_DIR}")
    print(f"  Results CSV: {RESULT_CSV}")

    print(f"\nProcessing Configuration:")
    print(f"  Target Organs ({len(CONFIG['organs'])}):")
    for organ in CONFIG['organs']:
        print(f"    - {organ}")
    print(f"  Modalities: {CONFIG['modalities']}")
    print(f"  DICOM Auto-Selection: {CONFIG.get('dicom_selection', {}).get('auto_select', True)}")

    print(f"\nDry run complete. No files were modified.")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PET-CT Radiomics Pipeline using TotalSegmentator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Run full pipeline
  python run_pipeline.py --dry-run          # Preview without processing
  python run_pipeline.py --version-report   # Show dependency versions
  python run_pipeline.py --input /path/to/dicom --output /path/to/output
        """
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview processing without executing")
    parser.add_argument("--version-report", action="store_true",
                        help="Print version information for all dependencies")
    parser.add_argument("--input", type=str, default=None,
                        help="Input DICOM directory (overrides config)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (overrides config)")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Configuration file path")
    parser.add_argument("--no-auto-select", action="store_true",
                        help="Disable DICOM series auto-selection")

    args = parser.parse_args()

    # 特殊モード
    if args.version_report:
        print_version_report()
        sys.exit(0)

    if args.dry_run:
        run_dry_run()
        sys.exit(0)

    # ディレクトリオーバーライド
    if args.input:
        DICOM_DIR = Path(args.input)
    if args.output:
        NIFTI_DIR = Path(args.output) / "nifti_images"
        SEG_DIR = Path(args.output) / "segmentations"
        NIFTI_DIR.mkdir(parents=True, exist_ok=True)
        SEG_DIR.mkdir(parents=True, exist_ok=True)

    # 自動選別の無効化
    if args.no_auto_select:
        CONFIG['dicom_selection']['auto_select'] = False

    log_progress("=== Pipeline Started ===", "INFO")

    # GPU確認
    gpu_available, gpu_message = check_gpu_memory()
    if gpu_available:
        log_progress(f"GPU Status: {gpu_message}", "INFO")
        use_gpu = True
    else:
        log_progress(f"GPU Status: {gpu_message}. Using CPU mode.", "WARNING")
        use_gpu = False

    all_results = []
    if not DICOM_DIR.exists():
        log_progress(f"Error: {DICOM_DIR} does not exist.", "ERROR")
        sys.exit(1)

    target_folders = [p.name for p in DICOM_DIR.iterdir() if p.is_dir()]
    log_progress(f"Found {len(target_folders)} patients. "
                f"Config: modalities={CONFIG['modalities']}, organs={CONFIG['organs']}", "INFO")

    for patient_idx, folder_name in enumerate(target_folders, 1):
        anon_id = get_or_create_anon_id(folder_name)
        log_progress(f"\n[Patient {patient_idx}/{len(target_folders)}] "
                    f"Processing: {folder_name} (ID: {anon_id})", "INFO")

        patient_path = DICOM_DIR / folder_name
        modality_paths = detect_folder_structure(patient_path, CONFIG)
        log_progress(f"  Found modalities: {list(modality_paths.keys())}", "INFO")

        ct_seg_dir = None
        ct_nifti = None

        # CT処理
        if 'CT' in modality_paths:
            ct_nifti = step1_convert_dicom(modality_paths['CT'], anon_id, 'CT')
            if ct_nifti:
                ct_seg_dir = step2_segmentation(ct_nifti, anon_id, 'CT', use_gpu=use_gpu)
                if ct_seg_dir:
                    feats = step3_radiomics(ct_nifti, ct_seg_dir, anon_id, 'CT')
                    all_results.extend(feats)

        # PET/他モダリティ処理
        for modality, dicom_path in modality_paths.items():
            if modality == 'CT':
                continue

            log_progress(f"  Processing {modality}...", "INFO")
            nifti = step1_convert_dicom(dicom_path, anon_id, modality)
            if not nifti:
                continue

            # PET/SPECTはCT空間にリサンプリング
            if modality in ['PET', 'PT', 'SPECT'] and ct_nifti and ct_nifti.exists():
                resampled_path = NIFTI_DIR / f"{anon_id}_{modality}_registered.nii.gz"
                if not resampled_path.exists():
                    if resample_pet_to_ct_space(nifti, ct_nifti, resampled_path):
                        nifti = resampled_path
                    else:
                        log_progress(f"  Warning: Could not resample {modality} to CT space", "WARNING")
                else:
                    log_progress(f"  {modality} registered file exists: {resampled_path.name}", "INFO")
                    nifti = resampled_path

            seg_dir = step2_segmentation(nifti, anon_id, modality, ct_seg_dir, use_gpu=use_gpu)
            if not seg_dir:
                continue
            feats = step3_radiomics(nifti, seg_dir, anon_id, modality)
            all_results.extend(feats)

    # 結果保存
    if all_results:
        if os.path.exists(RESULT_CSV):
            df_exist = pd.read_csv(RESULT_CSV)
            df_new = pd.DataFrame(all_results)
            df_final = pd.concat([df_exist, df_new], ignore_index=True)
        else:
            df_final = pd.DataFrame(all_results)
        df_final = df_final.drop_duplicates(subset=['PatientID', 'Modality', 'Organ'], keep='last')
        df_final.to_csv(RESULT_CSV, index=False)
        log_progress(f"\n=== Pipeline Completed === Saved {len(df_final)} rows to {RESULT_CSV}", "INFO")

        # 可視化
        log_progress("\n=== Step 4: Generating PET-CT Fusion Visualizations ===", "INFO")
        try:
            pet_cases = df_final[df_final['Modality'] == 'PET']['PatientID'].unique()
            if len(pet_cases) > 0:
                log_progress(f"Found {len(pet_cases)} cases with PET data for visualization", "INFO")
                try:
                    from visualize_mask_verification import create_mask_verification

                    for case_id in pet_cases:
                        log_progress(f"  Visualizing: {case_id}", "INFO")
                        try:
                            create_mask_verification(case_id)
                            log_progress(f"  {case_id} visualization complete", "INFO")
                        except Exception as e:
                            log_progress(f"  Error visualizing {case_id}: {e}", "ERROR")

                    log_progress(f"=== Visualization Completed ===", "INFO")
                except ImportError as e:
                    log_progress(f"Warning: Could not import visualization module: {e}", "WARNING")
            else:
                log_progress("No PET data found, skipping visualization", "INFO")
        except Exception as e:
            log_progress(f"Warning: Visualization step failed: {e}", "WARNING")
    else:
        log_progress("\n=== Pipeline Completed === No results to save.", "WARNING")
