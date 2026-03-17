#!/usr/bin/env python3
"""
Compatibility utility for manuscript-aligned PET SUV feature extraction.

This script rebuilds PET SUV images and PET radiomics rows from existing
pipeline outputs. It is intended for compatibility with older workflows;
`run_pipeline.py` already performs the same SUV conversion and extraction.
"""

import argparse
import csv
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import yaml
from radiomics import featureextractor

from suv_converter import SUVConverter


BASE_DIR = Path(os.environ.get("PET_PIPELINE_ROOT", Path(__file__).parent)).resolve()


def load_config(config_path=None):
    """Load configuration with manuscript-aligned defaults."""
    default_config = {
        'organs': [
            'liver',
            'spleen',
            'kidney_left',
            'kidney_right',
            'adrenal_gland_left',
            'adrenal_gland_right',
            'aorta',
            'vertebrae_L1',
        ],
        'radiomics': {
            'params_file': 'params.yaml',
            'extract_ct': False,
        },
        'output': {
            'csv_file': 'radiomics_results.csv',
        },
    }

    if config_path is None:
        config_path = BASE_DIR / "config.yaml"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f) or {}
        for key, value in user_config.items():
            if isinstance(value, dict):
                default_config.setdefault(key, {}).update(value)
            else:
                default_config[key] = value

    return default_config


def get_output_root(output_dir=None):
    """Resolve output root."""
    return Path(output_dir).resolve() if output_dir else BASE_DIR


def get_params_path(config):
    """Resolve params.yaml path."""
    params_path = Path(config.get('radiomics', {}).get('params_file', 'params.yaml'))
    if not params_path.is_absolute():
        params_path = BASE_DIR / params_path
    return params_path


def create_pet_extractor(config):
    """Create the PET extractor using params.yaml."""
    params_path = get_params_path(config)
    if not params_path.exists():
        raise FileNotFoundError(f"PyRadiomics parameter file not found: {params_path}")
    return featureextractor.RadiomicsFeatureExtractor(str(params_path))


def get_minimum_roi_size(config):
    """Read minimum ROI size from params.yaml."""
    params_path = get_params_path(config)
    with open(params_path, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f) or {}
    return int((params.get('setting', {}) or {}).get('minimumROISize', 0) or 0)


def count_mask_voxels(mask_path):
    """Count foreground voxels."""
    return int(np.count_nonzero(nib.load(str(mask_path)).get_fdata() > 0))


def get_case_ids(seg_root):
    """Return anonymized case IDs from segmentation output."""
    if not seg_root.exists():
        return []

    case_ids = set()
    for folder in seg_root.iterdir():
        if not folder.is_dir():
            continue
        if folder.name.endswith("_CT"):
            case_ids.add(folder.name[:-3])
        else:
            case_ids.add(folder.name)
    return sorted(case_ids)


def load_reverse_id_mapping(output_root):
    """Map anonymized IDs back to original source folder names."""
    mapping_file = output_root / "id_mapping.csv"
    reverse_mapping = {}
    if not mapping_file.exists():
        return reverse_mapping

    with open(mapping_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original = row.get("Original_Folder_Name")
            anon = row.get("Anonymized_ID")
            if original and anon:
                reverse_mapping[anon] = original
    return reverse_mapping


def find_pet_dicom_folder(source_root, original_folder_name):
    """Find the PET/PT DICOM series for the source folder."""
    patient_dir = source_root / original_folder_name
    if not patient_dir.exists():
        return None

    candidate_dirs = [patient_dir] + [p for p in patient_dir.iterdir() if p.is_dir()]
    for directory in candidate_dirs:
        for file_path in directory.iterdir():
            if not file_path.is_file() or file_path.name.startswith('.'):
                continue
            try:
                ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                if getattr(ds, 'Modality', '').upper() in {'PT', 'PET'}:
                    return directory
            except Exception:
                continue
    return None


def ensure_pet_suv(case_id, pet_dicom_dir, nifti_root):
    """Create SUV image from the registered PET image if needed."""
    suv_path = nifti_root / f"{case_id}_PET_SUV.nii.gz"
    if suv_path.exists():
        return suv_path

    source_candidates = [
        nifti_root / f"{case_id}_PET_registered.nii.gz",
        nifti_root / f"{case_id}_PET.nii.gz",
    ]
    source_path = next((path for path in source_candidates if path.exists()), None)
    if source_path is None:
        raise FileNotFoundError(f"No PET NIfTI found for {case_id}")

    converter = SUVConverter(pet_dicom_dir)
    pet_img = nib.load(str(source_path))
    suv_data = converter.convert_to_suv(pet_img.get_fdata())
    suv_img = nib.Nifti1Image(suv_data.astype(np.float32), pet_img.affine, pet_img.header)
    nib.save(suv_img, str(suv_path))
    return suv_path


def extract_pet_features(case_id, image_path, seg_dir, organs, extractor, min_roi_size):
    """Extract PET radiomics rows for one case."""
    rows = []
    for organ in organs:
        mask_path = seg_dir / f"{organ}.nii.gz"
        if not mask_path.exists():
            continue

        voxel_count = count_mask_voxels(mask_path)
        if voxel_count == 0 or voxel_count < min_roi_size:
            continue

        result = extractor.execute(str(image_path), str(mask_path))
        row = {"PatientID": case_id, "Modality": "PET", "Organ": organ}
        for key, value in result.items():
            if key.startswith("original_"):
                row[key] = value
        rows.append(row)
    return rows


def merge_results(output_csv, new_rows):
    """Merge refreshed PET rows into the output CSV."""
    df_new = pd.DataFrame(new_rows)
    if df_new.empty:
        return df_new

    if output_csv.exists():
        df_existing = pd.read_csv(output_csv)
        key_cols = ['PatientID', 'Modality', 'Organ']
        existing_keys = set(tuple(row) for row in df_new[key_cols].itertuples(index=False, name=None))
        keep_mask = [
            tuple(row) not in existing_keys
            for row in df_existing[key_cols].itertuples(index=False, name=None)
        ]
        df_existing = df_existing.loc[keep_mask]
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_csv(output_csv, index=False)
    return df_final


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild PET SUV images and PET radiomics rows from existing outputs."
    )
    parser.add_argument("--config", type=str, default=str(BASE_DIR / "config.yaml"),
                        help="Configuration file path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output root (defaults to repository root)")
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = get_output_root(args.output)
    source_root = BASE_DIR / "raw_download"
    nifti_root = output_root / "nifti_images"
    seg_root = output_root / "segmentations"
    output_csv = output_root / config['output']['csv_file']

    case_ids = get_case_ids(seg_root)
    if not case_ids:
        print("No segmentation outputs found.")
        return

    reverse_mapping = load_reverse_id_mapping(output_root)
    extractor = create_pet_extractor(config)
    min_roi_size = get_minimum_roi_size(config)
    organs = config['organs']

    all_rows = []
    for case_id in case_ids:
        original_folder = reverse_mapping.get(case_id)
        if not original_folder:
            print(f"Skipping {case_id}: missing reverse ID mapping")
            continue

        pet_dicom_dir = find_pet_dicom_folder(source_root, original_folder)
        if pet_dicom_dir is None:
            print(f"Skipping {case_id}: PET DICOM folder not found")
            continue

        seg_dir = seg_root / f"{case_id}_CT"
        if not seg_dir.exists():
            seg_dir = seg_root / case_id
        if not seg_dir.exists():
            print(f"Skipping {case_id}: segmentation directory not found")
            continue

        try:
            suv_path = ensure_pet_suv(case_id, pet_dicom_dir, nifti_root)
            rows = extract_pet_features(case_id, suv_path, seg_dir, organs, extractor, min_roi_size)
            all_rows.extend(rows)
            print(f"{case_id}: extracted {len(rows)} PET rows")
        except Exception as e:
            print(f"{case_id}: ERROR - {e}")

    if not all_rows:
        print("No PET rows were generated.")
        return

    df_final = merge_results(output_csv, all_rows)
    print(f"Saved: {output_csv}")
    print(f"Rows: {len(df_final)}")


if __name__ == "__main__":
    main()
