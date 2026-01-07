# Clinical Quality Control Checklist

This document provides guidance for validating pipeline outputs before using radiomics features in research or clinical applications.

## Pre-processing Checks

### 1. DICOM Integrity

- [ ] All DICOM files are readable
- [ ] Patient ID is correctly anonymized
- [ ] Modality tags are correct (CT, PT)
- [ ] Series descriptions match expected protocol

### 2. SUV Conversion Verification

| Check | Expected | Warning Signs |
|-------|----------|---------------|
| Liver SUVmean | 1.5-3.0 | < 0.5 or > 5.0 indicates conversion error |
| Blood pool (aorta) | 1.5-2.5 | Unusually high/low values |
| Background (muscle) | 0.5-1.5 | Negative values indicate error |

**Vendor-specific issues:**
- TOSHIBA/Canon: Check for ÷100 conversion applied
- Philips: Verify SUV scale factor used
- Siemens/GE: Confirm decay correction applied

---

## Segmentation Quality

### Visual Inspection Points

For each organ, verify in the QC visualization:

| Organ | Common Issues | What to Check |
|-------|---------------|---------------|
| Liver | Partial inclusion, excludes lesions | Complete coverage, no lung/heart overlap |
| Spleen | Confused with kidney/stomach | Correct anatomical location |
| Kidneys | Left/right swapped, includes adrenal | Bilateral symmetry, exclude pelvis |
| Adrenal glands | Too small, merged with kidney | Visible as separate structure |
| Aorta | Includes heart, truncated | Continuous from arch to bifurcation |
| Vertebrae | Wrong level labeled | Check against anatomical landmarks |

### Registration Accuracy

**Signs of registration failure:**
- Organ boundaries don't align between CT and PET
- SUV values appear shifted from anatomy
- Unusual patterns at organ edges

**Common causes:**
- Patient motion between scans
- Respiratory phase mismatch
- Metal artifact distortion

---

## Radiomics Feature Validation

### Physiological Plausibility

| Feature | Liver (PET) | Kidney (PET) | Aorta (PET) |
|---------|-------------|--------------|-------------|
| Mean SUV | 1.5-3.0 | 2.0-4.0 | 1.5-2.5 |
| Max SUV | 3.0-8.0 | 4.0-10.0 | 2.5-5.0 |
| Volume (ml) | 1000-2000 | 100-200 | 50-100 |
| Entropy | 3.5-4.5 | 3.5-4.5 | 3.0-4.0 |

### Red Flags

**Immediate investigation required:**
- Negative SUV values
- Volume = 0 or extremely small
- NaN or Inf in any feature
- Entropy = 0 (uniform region)

**Review recommended:**
- SUVmax > 20 (possible lesion or artifact)
- Volume deviates >50% from expected
- Shape features inconsistent with anatomy

---

## Batch Processing QC

### Statistical Outlier Detection

For each feature across patients:
1. Calculate median and IQR
2. Flag values outside 1.5 × IQR
3. Review flagged cases visually

### Consistency Checks

- [ ] Same number of features for all patients
- [ ] Same organs extracted for all patients
- [ ] No missing values (unless organ not present)
- [ ] Feature distributions match expected patterns

---

## Common Artifacts

### PET-specific

| Artifact | Appearance | Impact on Radiomics |
|----------|------------|---------------------|
| Metal artifact | Streaks, dark bands | Corrupted texture features |
| Respiratory motion | Blurring at diaphragm | Increased entropy, reduced contrast |
| Attenuation overcorrection | Bright rim at edges | Elevated edge features |
| Reconstruction artifact | Structured noise | Texture feature artifacts |

### CT-specific

| Artifact | Appearance | Impact on Segmentation |
|----------|------------|------------------------|
| Metal implant | Star pattern | Incorrect boundaries |
| Motion blur | Blurred edges | Over/under-segmentation |
| Beam hardening | Dark bands | Incorrect HU values |
| Truncation | Cut-off edges | Incomplete organ |

---

## Documentation Requirements

For each processed case, record:

1. **Patient metadata**
   - Anonymized ID
   - Scan date
   - Scanner vendor/model

2. **Processing parameters**
   - Pipeline version
   - Configuration used
   - Processing timestamp

3. **QC results**
   - Visual inspection passed/failed
   - Any organs excluded and why
   - Notes on artifacts or issues

4. **Output validation**
   - Number of features extracted
   - Any missing organs
   - Flagged outlier values

---

## Failure Modes and Solutions

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| No output CSV | Pipeline crash | Check error logs, verify input format |
| Missing organs | Segmentation failure | Review CT quality, try non-fast mode |
| Wrong SUV range | Conversion error | Verify vendor detection, check DICOM tags |
| Unrealistic features | Registration failure | Review CT-PET alignment |
| Empty mask warning | Organ too small in PET space | Expected for small structures, exclude from analysis |

---

## Minimum QC Protocol

For routine processing, at minimum:

1. **Spot-check** 10% of cases visually
2. **Statistical review** of all feature distributions
3. **Automatic flagging** of outliers (>3 SD from mean)
4. **Documentation** of any excluded cases

For publication or clinical use:

1. **100% visual review** of segmentation
2. **Expert validation** of SUV plausibility
3. **Cross-reference** with radiology reports
4. **Reproducibility check** (re-run random subset)
