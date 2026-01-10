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

# Project base directory
BASE_DIR = Path(__file__).parent

# Python executable path (auto-detect for environment)
def get_python_path():
    """Get Python path based on environment"""
    # Use currently running Python (conda environment compatible)
    return sys.executable

PYTHON_PATH = get_python_path()

# Colored output
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
    """Execute a script"""
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
            timeout=600  # 10 minute timeout
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
    """Check for new DICOM data (folder name = patient ID)"""
    dicom_dir = BASE_DIR / "raw_download"
    seg_dir = BASE_DIR / "segmentations"

    if not dicom_dir.exists():
        return [], []

    # Processed patients (exist in segmentations folder)
    processed = set()
    if seg_dir.exists():
        processed = {p.name for p in seg_dir.iterdir() if p.is_dir()}

    # Detect new folders
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
        description='PET/CT Radiomics Integrated Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--skip-visualization', action='store_true',
                        help='Skip visualization')
    parser.add_argument('--force', action='store_true',
                        help='Reprocess existing data')
    parser.add_argument('--visualize-only', action='store_true',
                        help='Run visualization only')
    args = parser.parse_args()

    start_time = time.time()

    print_header("PET/CT Radiomics Integrated Analysis Pipeline")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {BASE_DIR}")
    print(f"Python: {PYTHON_PATH}")

    # Visualization only mode
    if args.visualize_only:
        print_header("Visualization Only Mode")

        print_step(1, "Lung Segmentation Verification Images")
        run_script("visualize_lung_segmentation.py", "Lung segmentation images generated")

        print_step(2, "PET-CT Alignment Verification Images")
        run_script("visualize_ct_pet_seg.py", "PET-CT alignment images generated")

        print_step(3, "SUV Radiomics Result Graphs")
        run_script("plot_suv_results.py", "SUV result graphs generated")

        print_header("Complete")
        print(f"Results saved in analysis_results/ folder")
        return

    # Check for new data
    new_data, all_data = check_new_data()
    print(f"\nPatient folders: {len(all_data)}")
    for folder in all_data:
        status = "NEW" if folder in new_data else "processed"
        print(f"  - {folder} [{status}]")

    if new_data:
        print(f"\nNew data: {len(new_data)}")
    elif not args.force:
        print_warning("No new data. Use --force option to reprocess.")

    # Step 1: DICOM conversion / Segmentation / Radiomics
    print_step(1, "DICOM Conversion / Segmentation / Radiomics Extraction")
    if not run_script("run_pipeline.py", "Base pipeline complete"):
        print_error("Error in base pipeline")
        if not args.force:
            return

    # Step 2: SUV correction and Radiomics recalculation
    print_step(2, "SUV Correction and Radiomics Recalculation")
    if not run_script("create_final_suv.py", "SUV correction complete"):
        print_warning("Error in SUV correction (continuing)")

    # Step 3: Visualization
    if not args.skip_visualization:
        print_step(3, "Visualization / Report Generation")

        print("  3.1 Lung segmentation verification images...")
        run_script("visualize_lung_segmentation.py", "Lung segmentation images")

        print("  3.2 PET-CT alignment verification images...")
        run_script("visualize_ct_pet_seg.py", "PET-CT alignment images")

        print("  3.3 SUV Radiomics result graphs...")
        run_script("plot_suv_results.py", "SUV result graphs")

    # Complete
    elapsed = time.time() - start_time
    print_header("Analysis Complete")
    print(f"Processing time: {elapsed/60:.1f} minutes")
    print(f"\nOutput files:")
    print(f"  - pet_ct_radiomics_results.csv  (Radiomics features)")
    print(f"  - analysis_results/             (Visualization images)")
    print(f"  - nifti_images/                 (NIfTI images)")
    print(f"  - segmentations/                (Segmentation masks)")

if __name__ == "__main__":
    main()
