# PET-CT Radiomics Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-available-blue.svg)](https://www.docker.com/)

**Fully automated, reproducible whole-body PET radiomics extraction using TotalSegmentator-derived CT segmentation masks.**

This pipeline provides a standardized workflow for extracting radiomic features from PET/CT data, supporting:
- **104 anatomical structures** via TotalSegmentator deep learning segmentation
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

The pipeline generates a CSV file with radiomic features for each organ (representative 8-organ set):

| patient_id | modality | organ | original_firstorder_Mean | original_firstorder_Maximum | original_shape_VoxelVolume | ... |
|------------|----------|-------|--------------------------|-----------------------------|-----------------------------|-----|
| 0001 | PET | liver | 2.10 | 6.09 | 1286000 | ... |
| 0001 | PET | spleen | 1.81 | 3.54 | 181000 | ... |
| 0001 | PET | kidney_left | 2.76 | 6.25 | 160000 | ... |
| 0001 | PET | kidney_right | 2.85 | 6.50 | 152000 | ... |
| 0001 | PET | adrenal_gland_left | 1.50 | 2.84 | 4300 | ... |
| 0001 | PET | adrenal_gland_right | 0.77 | 1.62 | 2100 | ... |
| 0001 | PET | aorta | 1.72 | 4.00 | 67000 | ... |
| 0001 | PET | vertebrae_L1 | 1.87 | 4.50 | 38000 | ... |

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
    ├── 2. PET-to-CT spatial alignment (nibabel affine resampling)
    │       Note: NOT de novo registration - uses NIfTI world coordinates
    │
    ├── 3. Automatic SUV conversion (vendor-neutral)
    │
    ├── 4. CT segmentation (TotalSegmentator, 104 structures)
    │
    ├── 5. CT-derived masks applied to co-registered PET
    │
    └── 6. PyRadiomics feature extraction (IBSI-compliant, binWidth=0.25 SUV)
            │
            └── CSV output (107 features × organs × modalities)
```

## Features

- **Automated CT Segmentation**: TotalSegmentator-based organ segmentation (104 anatomical structures)
- **Vendor-neutral SUV Conversion**: Automatic handling of manufacturer-specific DICOM variations
- **IBSI-aligned Radiomics**: PyRadiomics feature extraction with standardized settings (see [IBSI Compliance](#ibsi-compliance))
- **Quality Control Visualization**: Automatic generation of CT/PET fusion images with segmentation overlays
- **Cross-platform**: Works on Linux, macOS, and Windows
- **GPU Acceleration**: CUDA support for fast segmentation (~90 seconds per case)
- **DICOM Series Auto-Selection**: Automatically selects appropriate CT/PET series from multi-series data (v2.0+)

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.10+ |
| RAM | 8GB | 16GB+ |
| GPU | - | NVIDIA (8GB+ VRAM) |
| Storage | 5GB | 10GB+ |

### Dependencies (Version-Locked)

**Important:** Specific versions are required to avoid compatibility issues:

- **PyTorch >= 2.2** (required for TotalSegmentator compatibility)
- **NumPy >= 1.26, < 2.0** (NumPy 2.x incompatible with PyTorch/TotalSegmentator)
- TotalSegmentator >= 2.0, < 3.0
- PyRadiomics >= 3.0, < 4.0
- SimpleITK >= 2.3
- pydicom >= 2.4
- nibabel >= 5.0
- dicom2nifti >= 2.4

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

### Option 3: Docker (Recommended for Reproducibility)

Docker provides a fully isolated environment with all dependencies pre-configured.

```bash
# Clone the repository
git clone https://github.com/haga-aki/pet-ct-radiomics-pipeline.git
cd pet-ct-radiomics-pipeline

# Build Docker image
docker build -t pet-ct-radiomics .

# Run with CPU
docker run -v /path/to/dicom:/data/input:ro -v /path/to/output:/data/output \
    pet-ct-radiomics python run_pipeline.py --input /data/input --output /data/output

# Run with GPU (requires NVIDIA Container Toolkit)
docker run --gpus all -v /path/to/dicom:/data/input:ro -v /path/to/output:/data/output \
    pet-ct-radiomics python run_pipeline.py --input /data/input --output /data/output
```

**Using Docker Compose (simplified):**

```bash
# Place DICOM data in ./input directory
mkdir -p input output
cp -r /path/to/dicom/* input/

# Run with CPU
docker compose run pet-radiomics

# Run with GPU
docker compose run pet-radiomics-gpu

# Results will be in ./output directory
```

**GPU Requirements:**
- NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

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

# Or use all 104 organs:
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

## IBSI Compliance

This pipeline uses [PyRadiomics](https://pyradiomics.readthedocs.io/), which implements feature definitions aligned with the [Image Biomarker Standardization Initiative (IBSI)](https://theibsi.github.io/). The following settings are used in `params.yaml`:

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Resampling** | None (native resolution) | Preserves original PET voxel size (~4mm) |
| **Intensity discretization** | binWidth = 0.25 SUV | Fixed bin width for texture features (Pfaehler et al. 2019) |
| **First-order features** | Continuous SUV values | No discretization for first-order statistics |
| **Distance (GLCM)** | 1 voxel | Standard neighborhood |
| **Symmetrical GLCM** | True | Standard symmetric matrix |
| **Force 2D** | False | 3D volumetric extraction |
| **Minimum ROI size** | 50 voxels | Ensures reliable texture features |

**Important notes:**
- Feature *definitions* follow IBSI standards via PyRadiomics
- Fixed bin width discretization (0.25 SUV) is applied for texture feature computation
- First-order features are computed on continuous SUV values without discretization
- No spatial resampling is applied to preserve native PET resolution
- See [`params.yaml`](params.yaml) for the complete configuration

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
├── config.yaml.example      # Configuration template
├── params.yaml              # PyRadiomics settings
├── docs/                    # Documentation
│   ├── pipeline_overview.md
│   ├── radiomics_feature_list.md
│   ├── config_reference.md
│   └── clinical_qc_checklist.md
├── examples/                # Example scripts and output
│   ├── example_output.csv   # Sample radiomics output (8-organ set)
│   └── lung_analysis/       # Application example: lung lobe analysis
│       ├── plot_suv_results.py        # (separate from main 8-organ workflow)
│       ├── visualize_lung_segmentation.py
│       ├── visualize_ct_pet_seg.py
│       └── visualize_pet_ct_fusion.py
├── environment.yml          # Conda environment (Linux/Mac)
├── environment_windows.yml  # Conda environment (Windows)
├── requirements.txt         # pip requirements
├── Dockerfile               # Docker image definition
├── docker-compose.yml       # Docker Compose configuration
├── CITATION.cff             # Citation metadata
└── LICENSE                  # MIT License
```

## Citation

If you use this pipeline in your research, please cite this repository.
A manuscript describing this work is in preparation.

```bibtex
@misc{haga2026petct,
  title={PET-CT Radiomics Pipeline},
  author={Haga, Akira and others},
  year={2026},
  note={Manuscript in preparation},
  howpublished={\url{https://github.com/haga-aki/pet-ct-radiomics-pipeline}}
}
```

*Citation will be updated upon publication.*

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

*Additional contributors may be added upon publication.*

## Acknowledgments

- TotalSegmentator developers for the excellent segmentation model
- PyRadiomics team for IBSI-compliant feature extraction
- Radiology technologists at Yokohama City University Hospital
