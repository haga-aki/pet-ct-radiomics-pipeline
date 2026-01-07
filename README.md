# PET-CT Radiomics Pipeline

Automated PET/CT radiomics feature extraction pipeline using TotalSegmentator for CT segmentation.

## Features

- **Automated CT Segmentation**: TotalSegmentator-based organ segmentation (117 anatomical structures)
- **Multi-vendor SUV Support**: Automatic SUV conversion for TOSHIBA/Canon, Philips, Siemens, and GE scanners
- **IBSI-compliant Radiomics**: PyRadiomics feature extraction (107 standardized features)
- **Cross-platform**: Works on Linux, macOS, and Windows
- **GPU Acceleration**: CUDA support for fast segmentation

## Pipeline Overview

```
DICOM Input
    │
    ├── 1. DICOM → NIfTI conversion
    │
    ├── 2. Rigid PET-CT registration (SimpleITK)
    │
    ├── 3. Vendor-specific SUV conversion
    │
    ├── 4. CT segmentation (TotalSegmentator)
    │
    ├── 5. Mask resampling to PET space
    │
    └── 6. PyRadiomics feature extraction
            │
            └── CSV output
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- ~5GB disk space for TotalSegmentator models

### Dependencies

- TotalSegmentator >= 2.0
- PyRadiomics
- SimpleITK
- PyTorch
- PyDICOM

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/pet-ct-radiomics-pipeline.git
cd pet-ct-radiomics-pipeline
```

### 2. Create conda environment

```bash
# Linux/macOS
conda env create -f environment.yml
conda activate pet_radiomics

# Windows
conda env create -f environment_windows.yml
conda activate pet_radiomics
```

Or using pip:

```bash
pip install -r requirements.txt
```

### 3. Configure the pipeline

```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your settings
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
# Launch GUI (macOS/Windows)
python gui_launcher.py
```

### Configuration

Edit `config.yaml` to customize:

```yaml
# Modalities to process
modalities:
  - CT
  - PET

# Target organs (representative set covering size/uptake/PV variations)
organs:
  # Large organs (stable)
  - liver
  - spleen
  # Medium organs (physiologic uptake)
  - kidney_left
  - kidney_right
  # Small organs (PV effect)
  - adrenal_gland_left
  - adrenal_gland_right
  # Reference (blood pool)
  - aorta
  # Bone marrow (representative)
  - vertebrae_L1

# Use 'all' for all 117 organs:
# organs:
#   - all

# Segmentation settings
segmentation:
  fast: true  # Use fast mode for quicker processing
  roi_subset: null  # null = all organs (recommended)

# Output settings
output:
  csv_file: radiomics_results.csv
  include_diagnostics: false
```

## Output

The pipeline generates a CSV file containing:

- **Patient ID**: Anonymized identifier
- **Organ**: Segmented structure name
- **First-order features**: Mean, median, variance, entropy, etc.
- **Shape features**: Volume, surface area, sphericity, etc.
- **Texture features**: GLCM, GLRLM, GLSZM, NGTDM features

## SUV Conversion

The pipeline automatically handles vendor-specific SUV calculations:

| Vendor | Method |
|--------|--------|
| Siemens | Standard SUV formula |
| GE | Private DICOM tags (0009,100d) |
| Philips | SUV Scale Factor (7053,1000) |
| TOSHIBA/Canon | Philips-compatible method |

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
├── environment.yml          # Conda environment (Linux/Mac)
├── environment_windows.yml  # Conda environment (Windows)
└── requirements.txt         # pip requirements
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{haga2026petct,
  title={Automated PET Radiomics Pipeline Using TotalSegmentator-derived CT Segmentation},
  author={Haga, Akira and Utsunomiya, Daisuke},
  journal={Japanese Journal of Radiology},
  year={2026}
}
```

## References

- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- [PyRadiomics](https://pyradiomics.readthedocs.io/)
- [IBSI](https://theibsi.github.io/) - Image Biomarker Standardisation Initiative

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

- Akira Haga, MD, PhD - Yokohama City University Hospital
- Daisuke Utsunomiya, MD, PhD - Yokohama City University Hospital
