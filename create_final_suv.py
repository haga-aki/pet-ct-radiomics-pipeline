#!/usr/bin/env python
"""
正しいSUV画像を作成してRadiomicsを再計算
メーカー別のSUV変換を自動で行う

複数症例対応版: segmentations/内の全患者を処理
"""
import os
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from radiomics import featureextractor

from suv_converter import SUVConverter

# Base directory - defaults to script location, can be overridden by environment variable
BASE_DIR = Path(os.environ.get("PET_PIPELINE_ROOT", Path(__file__).parent))
DICOM_DIR = BASE_DIR / "raw_download"
NIFTI_DIR = BASE_DIR / "nifti_images"
SEG_DIR = BASE_DIR / "segmentations"

# 対象臓器（推奨：代表臓器セット）
# 臓器サイズ・生理集積・PV問題を横断的にカバー
ORGANS = [
    # 大臓器・安定
    "liver",
    "spleen",
    # 中等度・生理集積あり
    "kidney_left",
    "kidney_right",
    # 小臓器・PV影響あり
    "adrenal_gland_left",
    "adrenal_gland_right",
    # 参照領域（blood pool）
    "aorta",
    # 骨髄（代表）
    "vertebrae_L1",
]


def get_patient_ids():
    """segmentationsフォルダから患者IDリストを取得"""
    if not SEG_DIR.exists():
        return []
    return [p.name for p in SEG_DIR.iterdir() if p.is_dir()]


def find_pet_dicom_folder(patient_id):
    """患者のPET DICOMフォルダを探す"""
    patient_dicom_dir = DICOM_DIR / patient_id
    if not patient_dicom_dir.exists():
        return None

    import pydicom
    for subdir in patient_dicom_dir.iterdir():
        if subdir.is_dir():
            for f in list(subdir.iterdir())[:3]:
                if f.is_file():
                    try:
                        ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                        if getattr(ds, 'Modality', '').upper() == 'PT':
                            return subdir
                    except:
                        continue
    return None


def process_patient(patient_id, extractor):
    """1患者のSUV変換とRadiomics抽出"""
    print(f"\n{'='*60}")
    print(f"Processing: {patient_id}")
    print(f"{'='*60}")

    results = []

    # パス設定
    pet_reg_path = NIFTI_DIR / f"{patient_id}_PET_registered.nii.gz"
    suv_output_path = NIFTI_DIR / f"{patient_id}_PET_SUV.nii.gz"
    ct_path = NIFTI_DIR / f"{patient_id}_CT.nii.gz"
    seg_dir = SEG_DIR / patient_id

    # PET画像があるか確認
    if not pet_reg_path.exists():
        print(f"  PET registered image not found: {pet_reg_path.name}")
        # CTのみの場合はCT Radiomicsだけ抽出
        if ct_path.exists():
            print("  Processing CT only...")
            for organ in ORGANS:
                mask_path = seg_dir / f"{organ}.nii.gz"
                if mask_path.exists():
                    try:
                        result = extractor.execute(str(ct_path), str(mask_path))
                        row = {"PatientID": patient_id, "Modality": "CT", "Organ": organ}
                        for k, v in result.items():
                            if k.startswith("original_"):
                                row[k] = v
                        results.append(row)
                    except Exception as e:
                        print(f"    {organ}: ERROR - {e}")
        return results

    # 1. SUV画像を作成（メーカー自動検出）
    print("\n【1. SUV画像作成】")

    pet_dicom_dir = find_pet_dicom_folder(patient_id)
    if pet_dicom_dir:
        try:
            converter = SUVConverter(pet_dicom_dir)
            converter.print_info()

            img = nib.load(pet_reg_path)
            data = img.get_fdata()
            suv_data = converter.convert_to_suv(data)

            print(f"  変換後: max={suv_data.max():.2f} (SUVbw)")

            suv_img = nib.Nifti1Image(suv_data.astype(np.float32), img.affine, img.header)
            nib.save(suv_img, suv_output_path)
            print(f"  保存: {suv_output_path.name}")
        except Exception as e:
            print(f"  SUV変換エラー: {e}")
            return results
    else:
        print(f"  PET DICOMフォルダが見つかりません")
        return results

    # 2. Radiomics抽出
    print("\n【2. Radiomics抽出】")

    # CT Radiomics
    if ct_path.exists():
        print("\n  CT Radiomics...")
        for organ in ORGANS:
            mask_path = seg_dir / f"{organ}.nii.gz"
            if mask_path.exists():
                try:
                    result = extractor.execute(str(ct_path), str(mask_path))
                    row = {"PatientID": patient_id, "Modality": "CT", "Organ": organ}
                    for k, v in result.items():
                        if k.startswith("original_"):
                            row[k] = v
                    results.append(row)
                    print(f"    {organ}: OK")
                except Exception as e:
                    print(f"    {organ}: ERROR - {e}")

    # PET SUV Radiomics
    print("\n  PET SUV Radiomics...")
    for organ in ORGANS:
        mask_path = seg_dir / f"{organ}.nii.gz"
        if mask_path.exists():
            try:
                result = extractor.execute(str(suv_output_path), str(mask_path))
                row = {"PatientID": patient_id, "Modality": "PET", "Organ": organ}
                for k, v in result.items():
                    if k.startswith("original_"):
                        row[k] = v
                results.append(row)

                mean_suv = result.get('original_firstorder_Mean', 0)
                max_suv = result.get('original_firstorder_Maximum', 0)
                print(f"    {organ}: Mean={mean_suv:.3f}, Max={max_suv:.3f}")
            except Exception as e:
                print(f"    {organ}: ERROR - {e}")

    return results


def main():
    print("=" * 70)
    print("SUV変換とRadiomics再計算 (複数症例対応)")
    print("=" * 70)

    # 患者リスト取得
    patient_ids = get_patient_ids()
    if not patient_ids:
        print("No patients found in segmentations folder")
        return

    print(f"\n対象患者数: {len(patient_ids)}")
    for pid in patient_ids:
        print(f"  - {pid}")

    # Radiomics抽出器
    extractor = featureextractor.RadiomicsFeatureExtractor()

    # 全患者を処理
    all_results = []
    for patient_id in sorted(patient_ids):
        results = process_patient(patient_id, extractor)
        all_results.extend(results)

    # 結果保存
    if all_results:
        df = pd.DataFrame(all_results)
        output_csv = BASE_DIR / "pet_ct_radiomics_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"\n{'='*70}")
        print(f"保存: {output_csv}")
        print(f"総レコード数: {len(df)}")

        # サマリー表示
        print(f"\n{'='*70}")
        print("PET SUV Radiomics サマリー")
        print(f"{'='*70}")

        df_pet = df[df['Modality'] == 'PET']
        organ_names = {
            'lung_upper_lobe_left': '左上葉',
            'lung_lower_lobe_left': '左下葉',
            'lung_upper_lobe_right': '右上葉',
            'lung_middle_lobe_right': '右中葉',
            'lung_lower_lobe_right': '右下葉'
        }

        for patient_id in df_pet['PatientID'].unique():
            print(f"\n【{patient_id}】")
            print(f"{'肺葉':<8} {'Mean SUV':>10} {'Max SUV':>10} {'Std SUV':>10}")
            print("-" * 45)

            df_patient = df_pet[df_pet['PatientID'] == patient_id]
            for _, row in df_patient.iterrows():
                name = organ_names.get(row['Organ'], row['Organ'])
                mean_suv = row['original_firstorder_Mean']
                max_suv = row['original_firstorder_Maximum']
                std_suv = np.sqrt(row['original_firstorder_Variance'])
                print(f"{name:<8} {mean_suv:>10.3f} {max_suv:>10.3f} {std_suv:>10.3f}")

    # 腫瘍情報と対側肺情報のCSVを作成
    create_tumor_summary_csv(patient_ids)

    print("\n完了!")


def create_tumor_summary_csv(patient_ids):
    """腫瘍位置と対側肺SUV情報のサマリーCSVを作成"""
    print(f"\n{'='*70}")
    print("腫瘍・対側肺サマリーCSV作成")
    print(f"{'='*70}")

    summary_data = []

    for patient_id in sorted(patient_ids):
        # PET SUV画像をロード
        pet_suv_path = NIFTI_DIR / f"{patient_id}_PET_SUV_registered.nii.gz"
        if not pet_suv_path.exists():
            pet_suv_path = NIFTI_DIR / f"{patient_id}_PET_SUV.nii.gz"
        if not pet_suv_path.exists():
            continue

        seg_dir = SEG_DIR / patient_id
        if not seg_dir.exists():
            continue

        try:
            pet_img = nib.load(pet_suv_path)
            pet_data = pet_img.get_fdata()

            # 各肺葉のSUV情報を取得
            lobe_stats = {}
            for organ in ORGANS:
                mask_path = seg_dir / f"{organ}.nii.gz"
                if mask_path.exists():
                    mask = nib.load(mask_path).get_fdata() > 0
                    if np.any(mask):
                        pet_in_mask = pet_data[mask]
                        side = 'left' if 'left' in organ else 'right'
                        lobe_stats[organ] = {
                            'side': side,
                            'suv_max': np.max(pet_in_mask),
                            'suv_mean': np.mean(pet_in_mask),
                            'suv_std': np.std(pet_in_mask),
                            'suv_median': np.median(pet_in_mask),
                        }

            if not lobe_stats:
                continue

            # 腫瘍側（SUV maxが最大の肺葉）を特定
            primary_lobe = max(lobe_stats.keys(), key=lambda x: lobe_stats[x]['suv_max'])
            primary_stats = lobe_stats[primary_lobe]
            tumor_side = primary_stats['side']

            # 対側肺の統計を計算
            contralateral_side = 'left' if tumor_side == 'right' else 'right'
            contra_values = []
            for organ, stats in lobe_stats.items():
                if stats['side'] == contralateral_side:
                    mask_path = seg_dir / f"{organ}.nii.gz"
                    mask = nib.load(mask_path).get_fdata() > 0
                    contra_values.extend(pet_data[mask].tolist())

            if contra_values:
                contra_values = np.array(contra_values)
                contra_stats = {
                    'suv_max': np.max(contra_values),
                    'suv_mean': np.mean(contra_values),
                    'suv_std': np.std(contra_values),
                    'suv_median': np.median(contra_values),
                    'suv_95percentile': np.percentile(contra_values, 95),
                }
            else:
                contra_stats = None

            # 肺葉名の変換
            lobe_name_map = {
                'lung_upper_lobe_left': 'Left Upper',
                'lung_lower_lobe_left': 'Left Lower',
                'lung_upper_lobe_right': 'Right Upper',
                'lung_middle_lobe_right': 'Right Middle',
                'lung_lower_lobe_right': 'Right Lower',
            }

            row = {
                'PatientID': patient_id,
                'TumorLobe': lobe_name_map.get(primary_lobe, primary_lobe),
                'TumorSide': 'Right' if tumor_side == 'right' else 'Left',
                'Tumor_SUVmax': primary_stats['suv_max'],
                'Tumor_SUVmean': primary_stats['suv_mean'],
                'Tumor_SUVstd': primary_stats['suv_std'],
                'ContralateralSide': 'Right' if contralateral_side == 'right' else 'Left',
                'Contralateral_SUVmax': contra_stats['suv_max'] if contra_stats else None,
                'Contralateral_SUVmean': contra_stats['suv_mean'] if contra_stats else None,
                'Contralateral_SUVstd': contra_stats['suv_std'] if contra_stats else None,
                'Contralateral_SUVmedian': contra_stats['suv_median'] if contra_stats else None,
                'Contralateral_SUV95percentile': contra_stats['suv_95percentile'] if contra_stats else None,
            }

            # 各肺葉の詳細を追加
            for organ in ORGANS:
                if organ in lobe_stats:
                    lobe_label = lobe_name_map.get(organ, organ).replace(' ', '_')
                    row[f'{lobe_label}_SUVmax'] = lobe_stats[organ]['suv_max']
                    row[f'{lobe_label}_SUVmean'] = lobe_stats[organ]['suv_mean']

            summary_data.append(row)

            print(f"  {patient_id}: Tumor={lobe_name_map.get(primary_lobe)} (SUVmax={primary_stats['suv_max']:.2f}), "
                  f"Contralateral SUVmean={contra_stats['suv_mean']:.2f}" if contra_stats else "")

        except Exception as e:
            print(f"  {patient_id}: ERROR - {e}")

    # CSV保存
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        output_path = BASE_DIR / "tumor_contralateral_summary.csv"
        df_summary.to_csv(output_path, index=False)
        print(f"\n保存: {output_path}")
        print(f"症例数: {len(df_summary)}")


if __name__ == "__main__":
    main()
