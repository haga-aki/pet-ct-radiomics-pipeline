import sys
import os
import csv
from pathlib import Path
from datetime import datetime
import dicom2nifti
from totalsegmentator.python_api import totalsegmentator
from radiomics import featureextractor
import pandas as pd
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import yaml

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. GPU features disabled.")

def load_config(config_path="config.yaml"):
    default_config = {
        'organs': ['lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right', 'heart', 'aorta', 'trachea', 'esophagus'],
        'modalities': ['CT'],
        'segmentation': {'tasks': {'CT': 'total', 'MR': 'total_mr', 'PET': 'use_ct_mask', 'SPECT': 'use_ct_mask'}, 'fast': True},
        'output': {'csv_file': 'radiomics_results.csv', 'include_diagnostics': False}
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
    """Get root directory - defaults to script location or current directory"""
    # Check for environment variable override
    if os.environ.get("PET_PIPELINE_ROOT"):
        return Path(os.environ["PET_PIPELINE_ROOT"])
    # Default to script directory
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
    """Check GPU availability and memory status"""
    if not TORCH_AVAILABLE:
        return False, "PyTorch not available"

    if not torch.cuda.is_available():
        return False, "CUDA not available"

    try:
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        gpu_mem_free = gpu_mem_total - gpu_mem_allocated

        print(f"\n=== GPU Status ===")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {gpu_mem_total:.2f} GB")
        print(f"Allocated: {gpu_mem_allocated:.2f} GB")
        print(f"Reserved: {gpu_mem_reserved:.2f} GB")
        print(f"Free: {gpu_mem_free:.2f} GB")

        if gpu_mem_free < 2.0:  # Less than 2GB free
            return False, f"Insufficient GPU memory (only {gpu_mem_free:.2f} GB free)"

        return True, f"GPU ready ({gpu_mem_free:.2f} GB available)"
    except Exception as e:
        return False, f"GPU check failed: {e}"

def log_progress(message, level="INFO"):
    """Print progress message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def get_or_create_anon_id(original_folder_name):
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

def detect_folder_structure(patient_path):
    modalities = CONFIG['modalities']
    found = {}
    for mod in modalities:
        mod_path = patient_path / mod
        if mod_path.exists() and mod_path.is_dir():
            found[mod] = mod_path
    if not found:
        found['CT'] = patient_path
    return found

def step1_convert_dicom(dicom_path, anon_id, modality):
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
        sample = output_folder / "lung_upper_lobe_left.nii.gz"
        if not sample.exists():
            # roi_subsetの取得
            roi_subset = seg_config.get('roi_subset', None)
            device = 'gpu' if use_gpu else 'cpu'  # TotalSegmentatorでは'gpu'を使用

            # 環境変数でGPU 0を指定
            if use_gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'

            if roi_subset:
                log_progress(f"{modality}: Running TotalSegmentator (task={task}, roi_subset={len(roi_subset)} organs, device={device})...", "INFO")
                totalsegmentator(input=nifti_path, output=output_folder, task=task,
                               fast=seg_config.get('fast', True), device=device, roi_subset=roi_subset)
            else:
                log_progress(f"{modality}: Running TotalSegmentator (task={task}, device={device})...", "INFO")
                totalsegmentator(input=nifti_path, output=output_folder, task=task,
                               fast=seg_config.get('fast', True), device=device)
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
    """Resample mask to match reference image space using SimpleITK"""
    try:
        mask_img = sitk.ReadImage(str(mask_path))
        ref_img = sitk.ReadImage(str(ref_image_path))

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor for labels
        resampler.SetDefaultPixelValue(0)

        resampled_mask = resampler.Execute(mask_img)
        sitk.WriteImage(resampled_mask, str(output_path))
        return True
    except Exception as e:
        print(f"    - Resample error: {e}")
        return False

def step3_radiomics(nifti_path, seg_folder, anon_id, modality):
    organs = CONFIG['organs']
    include_diag = CONFIG['output'].get('include_diagnostics', False)
    if 'all' in organs:
        files = list(seg_folder.glob("*.nii.gz"))
        organs = [p.stem.replace('.nii', '') for p in files if 'combined_' not in p.name]

    # Check if PET/SPECT modality needs resampling
    needs_resampling = modality in ['PET', 'PT', 'SPECT']

    log_progress(f"{modality}: Extracting radiomics features for {len(organs)} organs...", "INFO")
    extractor = featureextractor.RadiomicsFeatureExtractor()
    features_list = []
    for idx, organ in enumerate(organs, 1):
        mask_path = seg_folder / f"{organ}.nii.gz"
        if not mask_path.exists():
            continue
        try:
            # For PET/SPECT, resample mask to match image space
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
    log_progress(f"{modality}: Radiomics extraction completed ({len(features_list)}/{len(organs)} successful).", "INFO")
    return features_list

if __name__ == "__main__":
    log_progress("=== Pipeline Started ===", "INFO")

    # Check GPU availability
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
        sys.exit()
    target_folders = [p.name for p in DICOM_DIR.iterdir() if p.is_dir()]
    log_progress(f"Found {len(target_folders)} patients. Config: modalities={CONFIG['modalities']}, organs={CONFIG['organs']}", "INFO")

    for patient_idx, folder_name in enumerate(target_folders, 1):
        anon_id = get_or_create_anon_id(folder_name)
        log_progress(f"\n[Patient {patient_idx}/{len(target_folders)}] Processing: {folder_name} (ID: {anon_id})", "INFO")
        patient_path = DICOM_DIR / folder_name
        modality_paths = detect_folder_structure(patient_path)
        log_progress(f"  Found modalities: {list(modality_paths.keys())}", "INFO")
        ct_seg_dir = None
        if 'CT' in modality_paths:
            ct_nifti = step1_convert_dicom(modality_paths['CT'], anon_id, 'CT')
            if ct_nifti:
                ct_seg_dir = step2_segmentation(ct_nifti, anon_id, 'CT', use_gpu=use_gpu)
                if ct_seg_dir:
                    feats = step3_radiomics(ct_nifti, ct_seg_dir, anon_id, 'CT')
                    all_results.extend(feats)
        for modality, dicom_path in modality_paths.items():
            if modality == 'CT':
                continue
            log_progress(f"  Processing {modality}...", "INFO")
            nifti = step1_convert_dicom(dicom_path, anon_id, modality)
            if not nifti:
                continue
            seg_dir = step2_segmentation(nifti, anon_id, modality, ct_seg_dir, use_gpu=use_gpu)
            if not seg_dir:
                continue
            feats = step3_radiomics(nifti, seg_dir, anon_id, modality)
            all_results.extend(feats)
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

        # Step 4: Generate PET-CT fusion visualizations
        log_progress("\n=== Step 4: Generating PET-CT Fusion Visualizations ===", "INFO")
        try:
            # PETデータがある症例のリストを取得
            pet_cases = df_final[df_final['Modality'] == 'PET']['PatientID'].unique()
            if len(pet_cases) > 0:
                log_progress(f"Found {len(pet_cases)} cases with PET data for visualization", "INFO")

                # visualize_pet_ct_fusion.pyをインポートして実行
                try:
                    from visualize_pet_ct_fusion import create_pet_ct_fusion

                    for case_id in pet_cases:
                        log_progress(f"  Visualizing: {case_id}", "INFO")
                        try:
                            create_pet_ct_fusion(case_id)
                            log_progress(f"  ✓ {case_id} visualization complete", "INFO")
                        except Exception as e:
                            log_progress(f"  ✗ Error visualizing {case_id}: {e}", "ERROR")

                    log_progress(f"=== Visualization Completed === PNG files saved to visualizations/", "INFO")
                except ImportError as e:
                    log_progress(f"Warning: Could not import visualize_pet_ct_fusion.py: {e}", "WARNING")
            else:
                log_progress("No PET data found, skipping visualization", "INFO")
        except Exception as e:
            log_progress(f"Warning: Visualization step failed: {e}", "WARNING")
    else:
        log_progress("\n=== Pipeline Completed === No results to save.", "WARNING")
