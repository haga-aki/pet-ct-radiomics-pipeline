#!/usr/bin/env python3
"""
PET/CT Radiomics Pipeline - Full Analysis Script
================================================
Runs the manuscript-aligned pipeline and optional QC visualization.
"""

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).parent.resolve()
PYTHON_PATH = sys.executable


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.END}\n")


def print_step(step_num, text):
    print(f"{Colors.BLUE}{Colors.BOLD}[Step {step_num}]{Colors.END} {text}")


def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def get_output_root(output_dir=None):
    """Resolve the output root used by run_pipeline.py."""
    return Path(output_dir).resolve() if output_dir else BASE_DIR


def load_config(config_path):
    """Load config for reporting-only fields."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path
    if not config_path.exists():
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_id_mapping(output_root):
    """Load original-folder to anonymized-ID mapping."""
    id_map_file = output_root / "id_mapping.csv"
    mapping = {}
    if not id_map_file.exists():
        return mapping

    with open(id_map_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original = row.get("Original_Folder_Name")
            anon = row.get("Anonymized_ID")
            if original and anon:
                mapping[original] = anon
    return mapping


def get_processed_case_ids(output_root):
    """Collect processed anonymized IDs from segmentation directories."""
    seg_dir = output_root / "segmentations"
    if not seg_dir.exists():
        return set()

    case_ids = set()
    for folder in seg_dir.iterdir():
        if not folder.is_dir():
            continue
        name = folder.name
        if name.endswith("_CT"):
            case_ids.add(name[:-3])
        else:
            case_ids.add(name)
    return case_ids


def check_new_data(output_root, input_dir=None):
    """Check which source folders are not yet represented in processed outputs."""
    dicom_dir = Path(input_dir).resolve() if input_dir else BASE_DIR / "raw_download"
    if not dicom_dir.exists():
        return [], []

    mapping = load_id_mapping(output_root)
    processed_case_ids = get_processed_case_ids(output_root)
    processed_originals = {
        original for original, anon in mapping.items() if anon in processed_case_ids
    }

    all_folders = sorted([p.name for p in dicom_dir.iterdir() if p.is_dir()])
    new_folders = [folder for folder in all_folders if folder not in processed_originals]
    return new_folders, all_folders


def run_pipeline(args):
    """Run run_pipeline.py with the requested arguments."""
    cmd = [PYTHON_PATH, str(BASE_DIR / "run_pipeline.py"), "--config", args.config]
    if args.input:
        cmd.extend(["--input", args.input])
    if args.output:
        cmd.extend(["--output", args.output])

    result = subprocess.run(cmd, cwd=str(BASE_DIR), text=True)
    return result.returncode == 0


def run_visualization(output_root, case_ids):
    """Generate QC overlays for processed cases."""
    if not case_ids:
        print_warning("No processed cases found for visualization.")
        return True

    try:
        from visualize_mask_verification import create_mask_verification
    except ImportError as e:
        print_error(f"Visualization module import failed: {e}")
        return False

    nifti_dir = output_root / "nifti_images"
    seg_root = output_root / "segmentations"
    viz_root = output_root / "visualizations"

    success = True
    for case_id in case_ids:
        seg_dir = seg_root / f"{case_id}_CT"
        if not seg_dir.exists():
            seg_dir = seg_root / case_id
        if not seg_dir.exists():
            print_warning(f"Skipping visualization for {case_id}: segmentation not found")
            continue

        try:
            create_mask_verification(
                case_id,
                nifti_dir=nifti_dir,
                seg_dir=seg_dir,
                output_dir=viz_root / case_id,
            )
            print_success(f"Visualization generated for {case_id}")
        except Exception as e:
            print_error(f"Visualization failed for {case_id}: {e}")
            success = False

    return success


def main():
    parser = argparse.ArgumentParser(
        description="PET/CT Radiomics manuscript-aligned analysis runner"
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Configuration file path')
    parser.add_argument('--input', type=str, default=None,
                        help='Input DICOM directory (optional)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (optional)')
    parser.add_argument('--skip-visualization', action='store_true',
                        help='Skip QC visualization')
    parser.add_argument('--force', action='store_true',
                        help='Run pipeline even if no new source folders are detected')
    parser.add_argument('--visualize-only', action='store_true',
                        help='Generate visualization from existing outputs only')
    args = parser.parse_args()

    output_root = get_output_root(args.output)
    config = load_config(args.config)
    result_csv = output_root / config.get('output', {}).get('csv_file', 'radiomics_results.csv')
    start_time = time.time()

    print_header("PET/CT Radiomics Integrated Analysis Pipeline")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project directory: {BASE_DIR}")
    print(f"Output root: {output_root}")
    print(f"Python: {PYTHON_PATH}")

    if args.visualize_only:
        print_step(1, "QC visualization from existing outputs")
        case_ids = sorted(get_processed_case_ids(output_root))
        ok = run_visualization(output_root, case_ids)
        if ok:
            print_header("Complete")
        else:
            print_header("Complete with warnings")
        return

    new_data, all_data = check_new_data(output_root, args.input)
    print(f"\nPatient folders: {len(all_data)}")
    for folder in all_data:
        status = "NEW" if folder in new_data else "processed"
        print(f"  - {folder} [{status}]")

    if not new_data and not args.force:
        print_warning("No new data detected. Use --force to rerun the pipeline.")
        return

    print_step(1, "Run manuscript-aligned pipeline")
    if not run_pipeline(args):
        print_error("Pipeline execution failed")
        return
    print_success("Pipeline completed")

    if not args.skip_visualization:
        print_step(2, "Generate QC visualization")
        case_ids = sorted(get_processed_case_ids(output_root))
        run_visualization(output_root, case_ids)

    elapsed = time.time() - start_time
    print_header("Analysis Complete")
    print(f"Processing time: {elapsed / 60:.1f} minutes")
    print("\nOutput files:")
    print(f"  - {result_csv}")
    print(f"  - {output_root / 'nifti_images'}")
    print(f"  - {output_root / 'segmentations'}")
    print(f"  - {output_root / 'visualizations'}")


if __name__ == "__main__":
    main()
