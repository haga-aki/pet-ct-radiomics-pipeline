# Pipeline Overview

Detailed technical documentation of the PET-CT Radiomics Pipeline.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT LAYER                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  DICOM Directory                                                         │
│  ├── Patient_001/                                                        │
│  │   ├── CT/  (DICOM series)                                            │
│  │   └── PET/ (DICOM series)                                            │
│  └── Patient_002/                                                        │
│      └── ...                                                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING LAYER                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │ DICOM → NIfTI    │───▶│ PET-CT           │───▶│ SUV Conversion   │  │
│  │ Conversion       │    │ Registration     │    │ (Vendor-neutral) │  │
│  │ (dicom2nifti)    │    │ (SimpleITK)      │    │                  │  │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        SEGMENTATION LAYER                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    TotalSegmentator                               │   │
│  │                                                                   │   │
│  │  • Input: CT NIfTI                                               │   │
│  │  • Model: nnU-Net based (1.5GB)                                  │   │
│  │  • Output: 117 anatomical structure masks                        │   │
│  │  • Mode: fast (default) or full resolution                       │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Mask Resampling                                │   │
│  │                                                                   │   │
│  │  • Resample CT masks to PET space                                │   │
│  │  • Interpolation: Nearest-neighbor (preserve labels)             │   │
│  │  • Handle resolution mismatch                                    │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        EXTRACTION LAYER                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      PyRadiomics                                  │   │
│  │                                                                   │   │
│  │  For each organ × modality:                                      │   │
│  │  • First-order features (18)                                     │   │
│  │  • Shape features (14)                                           │   │
│  │  • GLCM features (24)                                            │   │
│  │  • GLRLM features (16)                                           │   │
│  │  • GLSZM features (16)                                           │   │
│  │  • GLDM features (14)                                            │   │
│  │  • NGTDM features (5)                                            │   │
│  │  ─────────────────                                               │   │
│  │  Total: 107 features                                             │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT LAYER                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │ radiomics.csv   │  │ QC Images       │  │ Intermediate    │         │
│  │                 │  │ (PNG)           │  │ NIfTI files     │         │
│  │ 107 features ×  │  │                 │  │                 │         │
│  │ organs ×        │  │ CT + mask       │  │ SUV images      │         │
│  │ modalities      │  │ PET-CT fusion   │  │ Resampled masks │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Processing

### Step 1: DICOM to NIfTI Conversion

**Input:** DICOM directory structure
**Output:** NIfTI files (.nii.gz)

```python
# Using dicom2nifti
dicom2nifti.convert_directory(dicom_dir, output_dir)
```

**Key considerations:**
- Automatically detects modality from DICOM tags
- Handles multi-series DICOM folders
- Preserves spatial orientation and metadata

---

### Step 2: PET-CT Registration

**Input:** CT and PET NIfTI files
**Output:** Registered PET in CT space

```python
# Using SimpleITK
registration = sitk.ImageRegistrationMethod()
registration.SetMetricAsMutualInformation()
registration.SetOptimizerAsGradientDescent()
transform = registration.Execute(fixed_ct, moving_pet)
```

**Parameters:**
- Metric: Mutual Information
- Optimizer: Gradient Descent
- Transform: Rigid (6 DOF)
- Interpolation: Linear for images

---

### Step 3: SUV Conversion

**Input:** Raw PET pixel values + DICOM metadata
**Output:** SUVbw values

The pipeline automatically detects the SUV encoding method from DICOM metadata:

#### Method 1: Pre-scaled SUV values

```python
# Detected via private DICOM tags
if has_prescaled_suv_tag:
    suv = pixel_value / scale_factor  # e.g., divide by 100
```

#### Method 2: Scale factor encoding

```python
# Scale factor stored in DICOM tag
suv = pixel_value * stored_scale_factor
```

#### Method 3: Activity concentration (BQML)

```python
# Standard SUV formula with decay correction
activity = pixel_value  # Bq/ml
weight = patient_weight  # g
dose = injected_dose  # Bq
decay = np.exp(-np.log(2) * time_elapsed / half_life)
suv = activity * weight / (dose * decay)
```

This vendor-neutral approach handles common DICOM variations automatically.

---

### Step 4: CT Segmentation

**Input:** CT NIfTI file
**Output:** 117 organ masks (NIfTI)

```bash
TotalSegmentator -i ct.nii.gz -o segmentations/ --fast
```

**Output structure:**
```
segmentations/
├── liver.nii.gz
├── spleen.nii.gz
├── kidney_left.nii.gz
├── kidney_right.nii.gz
├── ... (117 files)
└── combined_all_organs.nii.gz
```

---

### Step 5: Mask Resampling

**Input:** CT-space masks, PET reference image
**Output:** PET-space masks

```python
# Resample mask to PET space
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(pet_image)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Preserve labels
pet_mask = resampler.Execute(ct_mask)
```

**Important:**
- Must use nearest-neighbor interpolation
- Check for empty masks after resampling (small organs may have 0 voxels)

---

### Step 6: Radiomic Feature Extraction

**Input:** Image (CT or PET) + mask
**Output:** 107 features per organ

```python
from radiomics import featureextractor

extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.enableAllFeatures()

features = extractor.execute(image, mask)
```

**Feature categories:**
| Category | Features | Description |
|----------|----------|-------------|
| First-order | 18 | Intensity statistics |
| Shape | 14 | 3D morphology |
| GLCM | 24 | Texture co-occurrence |
| GLRLM | 16 | Run-length patterns |
| GLSZM | 16 | Size-zone patterns |
| GLDM | 14 | Dependence patterns |
| NGTDM | 5 | Neighborhood patterns |

---

## Data Flow

```
Patient DICOM
    │
    ├──▶ CT DICOM ──▶ CT NIfTI ──▶ TotalSegmentator ──▶ CT Masks
    │                    │                                  │
    │                    │                                  │
    │                    ▼                                  ▼
    │              Registration ◀───────────────── Resample to PET
    │                    │                                  │
    │                    │                                  │
    └──▶ PET DICOM ──▶ PET NIfTI ──▶ SUV Image ──▶ PET + Masks
                                         │              │
                                         │              │
                                         ▼              ▼
                                    PyRadiomics ◀──────┘
                                         │
                                         ▼
                                   radiomics.csv
```

---

## Performance Characteristics

| Step | Time (GPU) | Time (CPU) | Memory |
|------|------------|------------|--------|
| DICOM conversion | 5 sec | 5 sec | 1 GB |
| Registration | 10 sec | 30 sec | 2 GB |
| SUV conversion | 2 sec | 2 sec | 0.5 GB |
| Segmentation (fast) | 60 sec | 300 sec | 4 GB |
| Segmentation (full) | 180 sec | 600 sec | 8 GB |
| Mask resampling | 5 sec | 5 sec | 1 GB |
| Feature extraction | 30 sec | 30 sec | 2 GB |
| **Total** | **~2 min** | **~8 min** | **8 GB** |

---

## Error Handling

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "Cannot find DICOM files" | Wrong path or format | Verify directory structure |
| "CUDA out of memory" | GPU memory exceeded | Use `--fast` or CPU mode |
| "Empty mask" | Small organ in PET space | Skip organ or use larger ROI |
| "SUV values negative" | Conversion error | Check vendor detection |
| "Registration failed" | Large patient motion | Manual alignment or exclude |

### Logging

Pipeline generates logs at multiple levels:
```
logs/
├── pipeline.log        # Main workflow
├── segmentation.log    # TotalSegmentator output
├── extraction.log      # PyRadiomics details
└── errors.log          # Error messages only
```
