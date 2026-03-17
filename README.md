# PET-CT Radiomics Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-available-blue.svg)](https://www.docker.com/)

Automated PET radiomics extraction from integrated PET/CT studies using
TotalSegmentator-derived CT masks.

## What This Repository Implements

The public workflow is aligned with the manuscript pipeline:

1. Select volumetric CT and PET DICOM series automatically.
2. Convert DICOM to NIfTI with `dicom2nifti`.
3. Resample PET to CT space using the NIfTI affine transform.
4. Convert PET voxel values to SUVbw using PET DICOM metadata.
5. Generate CT masks with TotalSegmentator.
6. Extract PyRadiomics features from the resampled PET SUV image using the
   CT-derived masks.

Important clarification:

- This is not de novo PET/CT registration.
- PET is assumed to come from an integrated PET/CT acquisition.
- No additional spatial resampling is applied inside PyRadiomics after the
  PET image has been resampled to CT space.

## Scope

- Default output: PET radiomics rows for a representative 8-organ set
- Supported mask source: TotalSegmentator CT segmentation
- Available structures: up to 104 TotalSegmentator anatomical labels
- Feature families: first-order, shape, GLCM, GLRLM, GLSZM, GLDM, NGTDM

By default, the repository does not export a separate CT radiomics table.
Shape descriptors are still included because they are derived from the
CT-based segmentation mask geometry.

## Quick Start

```bash
git clone https://github.com/haga-aki/pet-ct-radiomics-pipeline.git
cd pet-ct-radiomics-pipeline
conda env create -f environment.yml
conda activate pet_radiomics
cp config.yaml.example config.yaml
python run_pipeline.py --input /path/to/PETCT_DICOM --output /path/to/output
```

Main output:

- `/path/to/output/radiomics_results.csv`
- `/path/to/output/nifti_images/`
- `/path/to/output/segmentations/`
- `/path/to/output/visualizations/`

## Representative Default Organs

The default configuration targets the following organs:

- liver
- spleen
- kidney_left
- kidney_right
- adrenal_gland_left
- adrenal_gland_right
- aorta
- vertebrae_L1

This subset matches the manuscript demonstration workflow. To process all
available structures, set:

```yaml
organs:
  - all
```

## PyRadiomics Settings

PET extraction uses [`params.yaml`](params.yaml). The manuscript-aligned
defaults are:

- `binWidth = 0.25`
- `resampledPixelSpacing = null`
- `normalize = false`
- `force2D = false`
- `minimumROISize = 50`

Interpretation:

- PET is first resampled to CT space by the pipeline.
- `resampledPixelSpacing = null` means PyRadiomics does not perform a second
  resampling step.
- Small ROIs below 50 voxels are skipped rather than exported as placeholder
  rows.

The feature set is described using IBSI terminology, but this repository does
not claim formal independent IBSI certification.

## Configuration

Copy [`config.yaml.example`](config.yaml.example) to `config.yaml` and adjust:

```yaml
modalities:
  - CT
  - PET

organs:
  - liver
  - spleen
  - kidney_left
  - kidney_right
  - adrenal_gland_left
  - adrenal_gland_right
  - aorta
  - vertebrae_L1

radiomics:
  params_file: params.yaml
  extract_ct: false
```

Notes:

- `CT` remains necessary because TotalSegmentator runs on CT.
- `extract_ct: false` preserves the manuscript-style PET-focused output.

Full configuration details are documented in
[`docs/config_reference.md`](docs/config_reference.md).

## Alternative Entry Points

- `python run_full_analysis.py`
  Runs `run_pipeline.py` and then generates QC overlays.
- `python create_final_suv.py`
  Compatibility utility that rebuilds PET SUV images and PET rows from existing
  outputs. This is mainly for older workflows; `run_pipeline.py` already does
  SUV conversion internally.
- `python gui_launcher.py`
  Starts the GUI launcher.

## Output Table

`radiomics_results.csv` contains one row per `(PatientID, Modality, Organ)`.

Typical columns include:

- `PatientID`
- `Modality`
- `Organ`
- `original_firstorder_Mean`
- `original_firstorder_Maximum`
- `original_shape_VoxelVolume`
- `original_glcm_*`
- `original_glrlm_*`
- `original_glszm_*`
- `original_gldm_*`
- `original_ngtdm_*`

## Quality Control

QC overlays are generated with [`visualize_mask_verification.py`](visualize_mask_verification.py).
These figures show PET/CT fusion images with representative mask overlays for
visual inspection of mask placement and PET/CT alignment.

## Data Handling

This repository is intended for code and documentation only.

- Raw DICOM data should remain outside version control.
- Anonymized ID mapping files are ignored by `.gitignore`.
- If you publish derived CSV files, ensure they do not contain patient-linked
  identifiers or institution-restricted data.

## Repository Layout

```text
pet-ct-radiomics-pipeline/
├── run_pipeline.py
├── run_full_analysis.py
├── create_final_suv.py
├── visualize_mask_verification.py
├── suv_converter.py
├── config.yaml.example
├── params.yaml
├── docs/
├── raw_download/        # ignored
├── nifti_images/        # ignored
├── segmentations/       # ignored
└── visualizations/      # ignored
```

## References

- Wasserthal J, et al. TotalSegmentator: robust segmentation of 104 anatomic
  structures in CT images. Radiology: Artificial Intelligence. 2023.
- van Griethuysen JJM, et al. Computational radiomics system to decode the
  radiographic phenotype. Cancer Research. 2017.
- Zwanenburg A, et al. The Image Biomarker Standardization Initiative. Radiology.
  2020.
- Pfaehler E, et al. Repeatability of [18F]FDG PET radiomic features. Medical
  Physics. 2019.
- Boellaard R. FDG PET/CT: EANM procedure guidelines for tumour imaging.
  Eur J Nucl Med Mol Imaging. 2015.

## License

MIT License.
