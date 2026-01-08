# Exception Handling Specification

This document defines the pipeline's behavior for edge cases and error conditions.

## Overview

The pipeline is designed to handle common edge cases gracefully, ensuring consistent output across diverse datasets. All exceptions are logged and documented in the output.

---

## ROI Size Handling

### Minimum Voxel Threshold

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `minimumROISize` | 50 voxels | Below this, texture features become unreliable |
| `minimumROIDimensions` | 2 voxels | Minimum extent in each dimension |

### Behavior

```
IF voxel_count < 50:
    - Skip feature extraction for this organ
    - Log warning: "Organ {name} has insufficient voxels ({count} < 50)"
    - Output row contains NaN for all features
    - "extraction_status" column = "insufficient_voxels"
```

### Common Causes

- **Small organs** (e.g., adrenal glands) after resampling to PET resolution
- **Partial FOV** where organ is cut off
- **Segmentation failure** resulting in very small mask

---

## Empty Mask Handling

### Definition

An empty mask has zero foreground voxels (label = 0 everywhere).

### Behavior

```
IF mask is empty (no voxels with label > 0):
    - Skip feature extraction
    - Log warning: "Empty mask for organ {name}"
    - Output row contains NaN for all features
    - "extraction_status" column = "empty_mask"
```

### Common Causes

- **Organ not present** in scan FOV
- **Segmentation failure** (model confidence too low)
- **Wrong label ID** specified in configuration

---

## Feature Computation Failures

### Uniform Intensity

Some features (e.g., GLCM, entropy) cannot be computed when all voxels have identical intensity.

```
IF all voxels have same intensity:
    - Affected features return NaN
    - Log info: "Uniform intensity in {organ}, some features set to NaN"
    - Other computable features are still extracted
```

### Division by Zero / Overflow

```
IF computation results in inf or NaN:
    - Feature value = NaN
    - Log warning with feature name and organ
```

---

## Registration Failures

### Detection

Registration failure is detected by:
- Optimizer did not converge
- Final metric value exceeds threshold
- Transform parameters are extreme (>100mm translation)

### Behavior

```
IF registration fails:
    - Log error: "Registration failed for case {id}"
    - Continue with unregistered images (if rigid alignment is acceptable)
    - OR skip case entirely (configurable)
    - "registration_status" column in output = "failed"
```

---

## SUV Conversion Failures

### Missing DICOM Tags

```
IF required DICOM tags missing:
    - Attempt fallback to standard formula
    - IF still missing critical data (weight, dose):
        - Log error: "Cannot compute SUV: missing {tag}"
        - Output raw activity values with warning
        - "suv_status" column = "raw_activity"
```

### Negative or Implausible Values

```
IF SUV < 0:
    - Log warning: "Negative SUV detected, likely conversion error"
    - Flag case for QC review

IF SUV_max > 50 (configurable threshold):
    - Log warning: "Unusually high SUV, verify conversion"
    - Flag case for QC review
```

---

## Output Specification

### CSV Columns for Exception Tracking

| Column | Type | Values |
|--------|------|--------|
| `extraction_status` | string | "success", "insufficient_voxels", "empty_mask", "error" |
| `voxel_count` | int | Number of voxels in resampled mask |
| `suv_status` | string | "converted", "raw_activity", "error" |
| `registration_status` | string | "success", "failed", "skipped" |
| `warnings` | string | Semicolon-separated warning messages |

### Example Output Row (Failed Case)

```csv
case_id,organ,extraction_status,voxel_count,original_firstorder_Mean,...
0001,adrenal_gland_right,insufficient_voxels,23,NaN,...
```

---

## Logging Levels

| Level | Use Case |
|-------|----------|
| INFO | Normal operation, case start/end |
| WARNING | Recoverable issues (small ROI, high SUV) |
| ERROR | Non-recoverable issues (missing files, crash) |

### Log File Location

```
output_directory/
├── radiomics_results.csv
├── pipeline.log           # All messages
└── errors.log             # ERROR level only
```

---

## Recommended QC Workflow

1. **Check `extraction_status` column** - Filter for non-"success" rows
2. **Review `warnings` column** - Identify potential issues
3. **Visual inspection** - Use generated overlay images for flagged cases
4. **Statistical outliers** - Flag values outside 3 SD from mean

---

## Configuration Options

These behaviors can be customized in `config.yaml`:

```yaml
exception_handling:
  # Minimum voxels for feature extraction
  min_voxel_count: 50

  # Action on empty mask: "skip", "error", "nan"
  empty_mask_action: "skip"

  # Action on registration failure: "continue", "skip", "error"
  registration_failure_action: "continue"

  # SUV plausibility thresholds
  suv_warning_max: 50
  suv_error_if_negative: true

  # Skip entire case on critical error
  fail_fast: false
```
