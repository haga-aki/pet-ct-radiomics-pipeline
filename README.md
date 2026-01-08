# PET-CT Radiomics Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Fully automated, reproducible whole-body PET radiomics extraction using TotalSegmentator-derived CT segmentation masks.**

This pipeline provides a standardized workflow for extracting radiomic features from PET/CT data, supporting:
- **117 anatomical structures** via TotalSegmentator deep learning segmentation
- **Vendor-neutral SUV conversion** with automatic DICOM metadata handling
- **107 IBSI-compliant radiomic features** via PyRadiomics

## Quick Start

```bash
# Clone and setup
git clone https://github.com/haga-aki/pet-ct-radiomics-pipeline.git
cd pet-ct-radiomics-pipeline
conda env create -f environment.yml
conda activate pet_radiomics

# Run pipeline
python run_pipeline.py --input /path/to/PETCT_DICOM --output /path/to/output
```

## Example Output

The pipeline generates a CSV file with radiomic features for each organ:

| patient_id | modality | organ | original_firstorder_Mean | original_firstorder_Maximum | original_shape_VoxelVolume | original_glcm_Correlation | ... |
|------------|----------|-------|--------------------------|-----------------------------|-----------------------------|---------------------------|-----|
| 0001 | PET | liver | 2.10 | 6.09 | 1286000 | 0.82 | ... |
| 0001 | PET | spleen | 1.81 | 3.54 | 181000 | 0.78 | ... |
| 0001 | PET | kidney_left | 2.76 | 6.25 | 160000 | 0.75 | ... |
| 0001 | PET | aorta | 1.72 | 4.00 | 67000 | 0.71 | ... |
| 0001 | CT | liver | 58.2 | 180.5 | 1286000 | 0.85 | ... |

**107 features per organ** including:
- First-order statistics (18): mean, median, entropy, energy, etc.
- Shape features (14): volume, surface area, sphericity, etc.
- Texture features (75): GLCM, GLRLM, GLSZM, GLDM, NGTDM

## Pipeline Overview

```
DICOM Input (PET + CT)
    │
    ├── 1. DICOM → NIfTI conversion (dicom2nifti)
    │
    ├── 2. Rigid PET-CT registration (SimpleITK)
    │
    ├── 3. Automatic SUV conversion (vendor-neutral)
    │
    ├── 4. CT segmentation (TotalSegmentator, 117 structures)
    │
    ├── 5. Mask resampling to PET space (nearest-neighbor)
    │
    └── 6. PyRadiomics feature extraction (IBSI-compliant)
            │
            └── CSV output (107 features × organs × modalities)
```

## Features

- **Automated CT Segmentation**: TotalSegmentator-based organ segmentation (117 anatomical structures)
- **Vendor-neutral SUV Conversion**: Automatic handling of manufacturer-specific DICOM variations
- **IBSI-compliant Radiomics**: PyRadiomics feature extraction (107 standardized features)
- **Quality Control Visualization**: Automatic generation of CT/PET fusion images with segmentation overlays
- **Cross-platform**: Works on Linux, macOS, and Windows
- **GPU Acceleration**: CUDA support for fast segmentation (~90 seconds per case)

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.10+ |
| RAM | 8GB | 16GB+ |
| GPU | - | NVIDIA (8GB+ VRAM) |
| Storage | 5GB | 10GB+ |

### Dependencies

- TotalSegmentator >= 2.0
- PyRadiomics >= 3.0
- SimpleITK >= 2.0
- PyTorch >= 2.0
- pydicom
- nibabel
- dicom2nifti

## Installation

### Option 1: Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/haga-aki/pet-ct-radiomics-pipeline.git
cd pet-ct-radiomics-pipeline

# Create conda environment
conda env create -f environment.yml  # Linux/macOS
# or
conda env create -f environment_windows.yml  # Windows

conda activate pet_radiomics

# Create configuration file
cp config.yaml.example config.yaml
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
# Run the full pipeline
python run_pipeline.py --input /path/to/dicom --output /path/to/output

# Run with specific configuration
python run_pipeline.py --config config.yaml
```

### GUI Application

```bash
python gui_launcher.py
```

### Configuration

Edit `config.yaml` to customize target organs and settings:

```yaml
# Target organs (representative set recommended for validation)
organs:
  - liver          # Large, stable
  - spleen         # Large, stable
  - kidney_left    # Medium, physiologic uptake
  - kidney_right   # Medium, physiologic uptake
  - adrenal_gland_left   # Small, PV-susceptible
  - adrenal_gland_right  # Small, PV-susceptible
  - aorta          # Blood pool reference
  - vertebrae_L1   # Bone marrow representative

# Or use all 117 organs:
# organs:
#   - all
```

See [docs/config_reference.md](docs/config_reference.md) for full configuration options.

## Output Specification

### CSV Columns

| Column | Description | Example |
|--------|-------------|---------|
| patient_id | Anonymized patient identifier | 0001 |
| modality | Image modality | PET, CT |
| organ | TotalSegmentator label | liver, spleen |
| original_firstorder_Mean | Mean intensity/SUV | 2.10 |
| original_firstorder_Maximum | Maximum intensity/SUV | 6.09 |
| original_shape_VoxelVolume | Volume in mm³ | 1286000 |
| original_glcm_* | GLCM texture features | 0.82 |
| original_glrlm_* | GLRLM run-length features | 0.45 |
| original_glszm_* | GLSZM size-zone features | 0.33 |
| original_gldm_* | GLDM dependence features | 0.28 |
| original_ngtdm_* | NGTDM coarseness features | 0.15 |

See [docs/radiomics_feature_list.md](docs/radiomics_feature_list.md) for complete feature definitions.

## SUV Conversion

The pipeline is designed to process vendor-neutral DICOM PET/CT data. SUV conversion is handled automatically by detecting the encoding method from DICOM metadata:

- **Pre-scaled SUV values**: Detected via private tags, converted appropriately
- **Activity concentration (BQML)**: Standard decay-corrected SUV formula applied
- **Scale factor encoding**: Multiplied by stored scale factor

The implementation handles common manufacturer-specific DICOM variations without requiring manual configuration.

## Quality Control

The pipeline generates visualization images for each organ:

- **Panel 1**: CT only (anatomical reference)
- **Panel 2**: CT + segmentation mask overlay
- **Panel 3**: PET-CT fusion + segmentation mask

These visualizations enable rapid verification of:
- Registration accuracy
- Segmentation quality
- SUV plausibility

## Example Data

This pipeline has been validated with:
- Clinical whole-body PET/CT scanner data
- Representative organ set (8 organs covering size/uptake variations)

For public PET/CT datasets compatible with this pipeline, see:
- [TCIA NSCLC-Radiomics](https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics)
- [TCIA Head-Neck-PET-CT](https://wiki.cancerimagingarchive.net/display/Public/Head-Neck-PET-CT)

## Project Structure

```
pet-ct-radiomics-pipeline/
├── run_pipeline.py          # Main pipeline script
├── run_full_analysis.py     # Complete analysis workflow
├── suv_converter.py         # Multi-vendor SUV conversion
├── create_final_suv.py      # SUV correction and radiomics
├── gui_launcher.py          # GUI application
├── visualize_*.py           # Visualization scripts
├── config.yaml.example      # Configuration template
├── docs/                    # Documentation
│   ├── pipeline_overview.md
│   ├── radiomics_feature_list.md
│   ├── config_reference.md
│   └── clinical_qc_checklist.md
├── environment.yml          # Conda environment (Linux/Mac)
├── environment_windows.yml  # Conda environment (Windows)
├── requirements.txt         # pip requirements
├── CITATION.cff             # Citation metadata
└── LICENSE                  # MIT License
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{haga2026petct,
  title={Automated Multi-Organ PET Radiomics Extraction Using TotalSegmentator-Derived CT Segmentation Masks},
  author={Haga, Akira and Utsunomiya, Daisuke},
  journal={Medical Physics},
  year={2026},
  note={Technical Note}
}
```

## References

- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) - Wasserthal et al., Radiol Artif Intell 2023
- [PyRadiomics](https://pyradiomics.readthedocs.io/) - van Griethuysen et al., Cancer Res 2017
- [IBSI](https://theibsi.github.io/) - Image Biomarker Standardisation Initiative

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Authors

- **Akira Haga, MD, PhD** - Yokohama City University Hospital
- **Daisuke Utsunomiya, MD, PhD** - Yokohama City University Hospital

## Acknowledgments

- TotalSegmentator developers for the excellent segmentation model
- PyRadiomics team for IBSI-compliant feature extraction
- Radiology technologists at Yokohama City University Hospital
