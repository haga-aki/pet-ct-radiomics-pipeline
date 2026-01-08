# PET/CT Radiomics Pipeline - Setup Guide

## Overview

This pipeline automates the following processes:
- DICOM to NIfTI conversion
- PET-to-CT spatial alignment (nibabel affine resampling)
- TotalSegmentator-based organ segmentation (104 structures)
- Multi-vendor SUV conversion
- PyRadiomics feature extraction
- Result visualization

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.10+ |
| RAM | 8GB | 16GB+ |
| GPU | - | NVIDIA (8GB+ VRAM) |
| Storage | 5GB | 10GB+ |

### Supported Operating Systems

- **Linux**: Ubuntu 20.04+, tested on Ubuntu 24.04
- **macOS**: Intel/Apple Silicon (M1/M2/M3)
- **Windows**: Windows 10/11

---

## Installation

### Option 1: Conda (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/pet-ct-radiomics-pipeline.git
cd pet-ct-radiomics-pipeline

# 2. Create conda environment
# Linux/macOS
conda env create -f environment.yml
conda activate pet_radiomics

# Windows
conda env create -f environment_windows.yml
conda activate pet_radiomics

# 3. Create configuration file
cp config.yaml.example config.yaml
```

### Option 2: pip

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/pet-ct-radiomics-pipeline.git
cd pet-ct-radiomics-pipeline

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install CUDA-enabled PyTorch
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Create configuration file
cp config.yaml.example config.yaml
```

### Windows Quick Setup

```cmd
REM Run the setup script
setup_windows.bat
```

This script will:
- Create conda environment (`pet_radiomics`)
- Install all dependencies
- Optionally install CUDA-enabled PyTorch
- Create config.yaml

---

## Directory Structure

```
pet-ct-radiomics-pipeline/
├── raw_download/           # Place DICOM data here
│   ├── Patient_001/
│   │   ├── CT/            # CT DICOM series
│   │   └── PET/           # PET DICOM series
│   └── Patient_002/
│       └── ...
├── nifti_images/           # Converted NIfTI (auto-generated)
├── segmentations/          # Segmentation results (auto-generated)
├── config.yaml             # Configuration file
├── config.yaml.example     # Configuration template
├── run_pipeline.py         # Main pipeline script
└── radiomics_results.csv   # Output (auto-generated)
```

---

## Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
# Modalities to process
modalities:
  - CT
  - PET

# Target organs (lung lobes)
organs:
  - lung_upper_lobe_left
  - lung_lower_lobe_left
  - lung_upper_lobe_right
  - lung_middle_lobe_right
  - lung_lower_lobe_right

# Segmentation settings
segmentation:
  fast: true              # Fast mode (lower resolution)
  tasks:
    CT: total             # Full organ segmentation
    MR: total_mr          # MR-specific task
    PET: use_ct_mask      # Use CT mask for PET
    SPECT: use_ct_mask    # Use CT mask for SPECT
  roi_subset:             # Limit to thoracic organs
    - lung_upper_lobe_left
    - lung_lower_lobe_left
    - lung_upper_lobe_right
    - lung_middle_lobe_right
    - lung_lower_lobe_right
    - heart
    - aorta

# Output settings
output:
  csv_file: radiomics_results.csv
  include_diagnostics: false
```

---

## Usage

### Basic Usage

```bash
# Activate environment
conda activate pet_radiomics

# Run pipeline
python run_pipeline.py
```

### GUI Application

```bash
# Launch GUI (macOS/Windows)
python gui_launcher.py
```

### Batch Scripts

```bash
# macOS
./run_analysis.command

# Windows
run_analysis.bat
```

---

## Processing Steps

1. **Data Preparation**: Place DICOM folders in `raw_download/`
2. **ID Anonymization**: Folder names are converted to anonymous IDs
3. **DICOM to NIfTI**: Automatic conversion with modality detection
4. **Segmentation**: TotalSegmentator processes CT images (104 structures)
5. **Spatial Alignment**: PET resampled to CT space via nibabel affine (NOT de novo registration)
6. **SUV Conversion**: Vendor-neutral SUV calculation
7. **Feature Extraction**: 107 IBSI-compliant radiomics features
8. **Output**: Results saved to CSV

---

## Supported PET Vendors

The pipeline automatically handles vendor-specific SUV conversion:

| Vendor | Detection | Conversion Method |
|--------|-----------|-------------------|
| TOSHIBA/Canon | Tag (7065,102D) | Pixel / 100 |
| Philips | Tag (7053,1000) | SUVScaleFactor |
| Siemens | Bq/ml units | Decay correction |
| GE | Bq/ml units | Decay correction |

---

## Output Files

| File | Description |
|------|-------------|
| `radiomics_results.csv` | Radiomics features (107 features × organs × modalities) |
| `nifti_images/*.nii.gz` | Converted NIfTI images |
| `segmentations/*/` | Organ segmentation masks |
| `segmentations/*/combined_all_organs.nii.gz` | Combined mask for visualization |

---

## Troubleshooting

### GPU Issues with TotalSegmentator

```bash
# Run in CPU mode
export TOTALSEG_DISABLE_GPU=1  # Linux/Mac
set TOTALSEG_DISABLE_GPU=1     # Windows
python run_pipeline.py
```

### Out of Memory

- Set `fast: true` in config.yaml (uses lower resolution)
- Reduce `batch_size` in segmentation settings

### DICOM Not Detected

- Verify folder structure matches expected format
- DICOM files can be with or without `.dcm` extension
- Ensure modality-specific subfolders (CT/, PET/) exist

### TotalSegmentator Model Download

Models (~1.5GB) are downloaded automatically on first run.
If download fails, check internet connection and try again.

---

## Apple Silicon (M1/M2/M3) Notes

- TotalSegmentator runs in CPU mode (MPS not supported)
- Processing is slower but stable
- Use `fast: true` for reasonable performance

---

## License

This pipeline is released under the MIT License.

Note: TotalSegmentator is free for academic use only. Commercial use requires a license from the developers.

---

## References

- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) - Wasserthal et al.
- [PyRadiomics](https://pyradiomics.readthedocs.io/) - van Griethuysen et al.
- [IBSI](https://theibsi.github.io/) - Image Biomarker Standardisation Initiative
