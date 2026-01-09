# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-01-09

### Added
- **DICOM Series Auto-Selection**: Automatically detects and selects appropriate CT/PET series from multi-series DICOM data
  - Filters out Dose Reports, Localizers, MIP images, and other non-volumetric series
  - Configurable minimum slice counts (`min_ct_slices`, `min_pet_slices`)
  - Series selection based on SeriesDescription, slice count, and image dimensions
- **New visualization module** (`visualize_mask_verification.py`):
  - Mediastinal window CT settings (WL=40, WW=400)
  - PET-CT fusion with mask overlay
  - Axial + Coronal dual-view display
  - Automatic slice selection based on organ positions
- **Version report command**: `python run_pipeline.py --version-report`
- **Dry run mode**: `python run_pipeline.py --dry-run` for previewing without processing

### Changed
- **Updated default organ set**:
  - Liver, bilateral kidneys, bilateral adrenal glands, L1 vertebra, aorta, left lung (upper/lower lobes)
  - Optimized for clinical PET/CT analysis
- **Dependency version locking**:
  - PyTorch >= 2.4 (required for TotalSegmentator compatibility)
  - NumPy >= 1.26, < 2.0 (NumPy 2.x incompatible)
  - TotalSegmentator >= 2.0, < 3.0
  - PyRadiomics >= 3.0, < 4.0
- **Environment files updated** for all platforms (macOS, Linux, Windows)

### Fixed
- **PyTorch/TotalSegmentator compatibility**: Fixed `ImportError: cannot import name 'DiagnosticOptions' from 'torch.onnx._internal.exporter'`
- **NumPy version conflict**: Fixed `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`
- **DICOM series structure recognition**: Fixed issue where multi-series folders were incorrectly processed as separate patients

### Technical Details

#### DICOM Auto-Selection Logic
When processing a folder with multiple DICOM series:
1. Scans all subfolders for DICOM files
2. Reads metadata from first file in each series (Modality, SeriesDescription, Rows, Columns)
3. Filters by minimum slice count (default: CT ≥ 100, PET ≥ 50)
4. For CT: Prioritizes 512x512 standard resolution
5. For PET: Prioritizes series with "Axial" or "Body" in description
6. Falls back to series with maximum slice count

#### Visualization Settings
- CT Window: Mediastinal (WL=40 HU, WW=400 HU)
- PET Color Map: Hot
- PET SUV Range: 0-10
- Mask Overlay Alpha: 0.3

## [1.0.0] - 2026-01-01

### Initial Release
- DICOM to NIfTI conversion
- TotalSegmentator-based segmentation
- PyRadiomics feature extraction
- Multi-vendor SUV conversion
- GUI launcher
- Docker support
