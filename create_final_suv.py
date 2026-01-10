#!/usr/bin/env python
"""
PET/CT Radiomics Pipeline - SUV Conversion and Feature Extraction

Creates SUV-corrected PET images and extracts radiomic features using
vendor-neutral SUV conversion. Processes all patients in segmentations/.
"""
import os
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from radiomics import featureextractor
import yaml

from suv_converter import SUVConverter

# Base directory - defaults to script location, can be overridden by environment variable
BASE_DIR = Path(os.environ.get("PET_PIPELINE_ROOT", Path(__file__).parent))
DICOM_DIR = BASE_DIR / "raw_download"
NIFTI_DIR = BASE_DIR / "nifti_images"
SEG_DIR = BASE_DIR / "segmentations"


def load_config(config_path=None):
    """Load configuration from config.yaml"""
    if config_path is None:
        config_path = BASE_DIR / "config.yaml"

    # Default organs (representative 8-organ set)
    default_organs = [
        "liver",
        "spleen",
        "kidney_left",
        "kidney_right",
        "adrenal_gland_left",
        "adrenal_gland_right",
        "aorta",
        "vertebrae_L1",
    ]

    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if config and 'organs' in config:
                return config['organs']

    return default_organs


# Load organs from config
ORGANS = load_config()


def get_patient_ids():
    """Get list of patient IDs from segmentations folder"""
    if not SEG_DIR.exists():
        return []
    return [p.name for p in SEG_DIR.iterdir() if p.is_dir()]


def find_pet_dicom_folder(patient_id):
    """Find the PET DICOM folder for a patient"""
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
    """Process SUV conversion and radiomics extraction for one patient"""
    print(f"\n{'='*60}")
    print(f"Processing: {patient_id}")
    print(f"{'='*60}")

    results = []

    # パス設定
    pet_reg_path = NIFTI_DIR / f"{patient_id}_PET_registered.nii.gz"
    suv_output_path = NIFTI_DIR / f"{patient_id}_PET_SUV.nii.gz"
    ct_path = NIFTI_DIR / f"{patient_id}_CT.nii.gz"
    seg_dir = SEG_DIR / patient_id

    # セグメンテーションディレクトリがない場合は_CT付きを試す
    if not seg_dir.exists():
        seg_dir = SEG_DIR / f"{patient_id}_CT"

    if not seg_dir.exists():
        print(f"  Segmentation directory not found for {patient_id}")
        return results

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
    print("PET/CT Radiomics - SUV Conversion and Feature Extraction")
    print("=" * 70)
    print(f"Target organs ({len(ORGANS)}): {ORGANS}")

    # Get patient list
    patient_ids = get_patient_ids()
    if not patient_ids:
        print("No patients found in segmentations folder")
        return

    print(f"\nPatients to process: {len(patient_ids)}")
    for pid in patient_ids:
        print(f"  - {pid}")

    # Radiomics extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()

    # Process all patients
    all_results = []
    for patient_id in sorted(patient_ids):
        results = process_patient(patient_id, extractor)
        all_results.extend(results)

    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        output_csv = BASE_DIR / "pet_ct_radiomics_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"\n{'='*70}")
        print(f"Saved: {output_csv}")
        print(f"Total records: {len(df)}")

        # Summary display
        print(f"\n{'='*70}")
        print("PET SUV Radiomics Summary")
        print(f"{'='*70}")

        df_pet = df[df['Modality'] == 'PET']
        for patient_id in df_pet['PatientID'].unique():
            print(f"\n[{patient_id}]")
            print(f"{'Organ':<25} {'Mean SUV':>10} {'Max SUV':>10} {'Std SUV':>10}")
            print("-" * 60)

            df_patient = df_pet[df_pet['PatientID'] == patient_id]
            for _, row in df_patient.iterrows():
                organ = row['Organ']
                mean_suv = row['original_firstorder_Mean']
                max_suv = row['original_firstorder_Maximum']
                std_suv = np.sqrt(row['original_firstorder_Variance'])
                print(f"{organ:<25} {mean_suv:>10.3f} {max_suv:>10.3f} {std_suv:>10.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
