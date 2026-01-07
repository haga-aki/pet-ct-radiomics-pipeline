# Configuration Reference

Complete reference for `config.yaml` settings.

## Basic Structure

```yaml
modalities:
  - CT
  - PET

organs:
  - liver
  - spleen
  # ... more organs

segmentation:
  fast: true
  tasks:
    CT: total
    PET: use_ct_mask

output:
  csv_file: radiomics_results.csv
  include_diagnostics: false
```

---

## Modalities

Specify which imaging modalities to process.

```yaml
modalities:
  - CT    # Computed Tomography
  - PET   # Positron Emission Tomography
  # - MR  # Magnetic Resonance (requires total_mr task)
  # - SPECT  # Single Photon Emission CT
```

---

## Organs

### Representative Organ Set (Recommended)

A balanced set covering different organ characteristics:

```yaml
organs:
  # Large organs - stable segmentation
  - liver
  - spleen

  # Medium organs - physiologic uptake
  - kidney_left
  - kidney_right

  # Small organs - partial volume susceptible
  - adrenal_gland_left
  - adrenal_gland_right

  # Reference regions
  - aorta          # Blood pool
  - vertebrae_L1   # Bone marrow
```

### All Organs

Use all 117 TotalSegmentator structures:

```yaml
organs:
  - all
```

### Complete Organ List (117 structures)

```yaml
# === SPLEEN ===
# - spleen

# === KIDNEYS ===
# - kidney_right
# - kidney_left

# === GALLBLADDER ===
# - gallbladder

# === LIVER ===
# - liver

# === STOMACH ===
# - stomach

# === PANCREAS ===
# - pancreas

# === ADRENAL GLANDS ===
# - adrenal_gland_right
# - adrenal_gland_left

# === LUNGS ===
# - lung_upper_lobe_left
# - lung_lower_lobe_left
# - lung_upper_lobe_right
# - lung_middle_lobe_right
# - lung_lower_lobe_right

# === ESOPHAGUS ===
# - esophagus

# === TRACHEA ===
# - trachea

# === THYROID ===
# - thyroid_gland

# === SMALL BOWEL ===
# - small_bowel

# === DUODENUM ===
# - duodenum

# === COLON ===
# - colon

# === URINARY BLADDER ===
# - urinary_bladder

# === PROSTATE ===
# - prostate

# === SACRUM ===
# - sacrum

# === VERTEBRAE ===
# - vertebrae_S1
# - vertebrae_L5
# - vertebrae_L4
# - vertebrae_L3
# - vertebrae_L2
# - vertebrae_L1
# - vertebrae_T12
# - vertebrae_T11
# - vertebrae_T10
# - vertebrae_T9
# - vertebrae_T8
# - vertebrae_T7
# - vertebrae_T6
# - vertebrae_T5
# - vertebrae_T4
# - vertebrae_T3
# - vertebrae_T2
# - vertebrae_T1
# - vertebrae_C7
# - vertebrae_C6
# - vertebrae_C5
# - vertebrae_C4
# - vertebrae_C3
# - vertebrae_C2
# - vertebrae_C1

# === HEART ===
# - heart
# - heart_myocardium
# - heart_atrium_left
# - heart_ventricle_left
# - heart_atrium_right
# - heart_ventricle_right

# === AORTA ===
# - aorta

# === INFERIOR VENA CAVA ===
# - inferior_vena_cava

# === PORTAL/SPLENIC VEIN ===
# - portal_vein_and_splenic_vein

# === PULMONARY ARTERIES ===
# - pulmonary_artery

# === ILIAC ARTERIES ===
# - iliac_artery_left
# - iliac_artery_right

# === ILIAC VEINS ===
# - iliac_vena_left
# - iliac_vena_right

# === HUMERUS ===
# - humerus_left
# - humerus_right

# === SCAPULA ===
# - scapula_left
# - scapula_right

# === CLAVICULA ===
# - clavicula_left
# - clavicula_right

# === FEMUR ===
# - femur_left
# - femur_right

# === HIP ===
# - hip_left
# - hip_right

# === RIBS ===
# - rib_left_1
# - rib_left_2
# - rib_left_3
# - rib_left_4
# - rib_left_5
# - rib_left_6
# - rib_left_7
# - rib_left_8
# - rib_left_9
# - rib_left_10
# - rib_left_11
# - rib_left_12
# - rib_right_1
# - rib_right_2
# - rib_right_3
# - rib_right_4
# - rib_right_5
# - rib_right_6
# - rib_right_7
# - rib_right_8
# - rib_right_9
# - rib_right_10
# - rib_right_11
# - rib_right_12

# === GLUTEUS MUSCLES ===
# - gluteus_maximus_left
# - gluteus_maximus_right
# - gluteus_medius_left
# - gluteus_medius_right
# - gluteus_minimus_left
# - gluteus_minimus_right

# === AUTOCHTHON MUSCLES ===
# - autochthon_left
# - autochthon_right

# === ILIOPSOAS MUSCLES ===
# - iliopsoas_left
# - iliopsoas_right

# === BRAIN ===
# - brain

# === SKULL ===
# - skull

# === FACE ===
# - face
```

---

## Segmentation Settings

```yaml
segmentation:
  # TotalSegmentator task per modality
  tasks:
    CT: total       # Standard CT segmentation
    MR: total_mr    # MR-specific model
    PET: use_ct_mask   # Use CT mask for PET
    SPECT: use_ct_mask # Use CT mask for SPECT

  # Fast mode (lower resolution, faster processing)
  fast: true  # Recommended for routine use

  # Subset of organs to segment (null = all 117)
  roi_subset: null
```

### Fast Mode

| Mode | Resolution | Speed | Accuracy |
|------|------------|-------|----------|
| fast: true | Lower | ~90 sec | Good |
| fast: false | Full | ~3 min | Best |

---

## Output Settings

```yaml
output:
  # Output CSV filename
  csv_file: radiomics_results.csv

  # Include PyRadiomics diagnostic features
  include_diagnostics: false
```

### Diagnostic Features

When `include_diagnostics: true`, additional columns are added:
- Image hash values
- Mask statistics
- Interpolation information
- Processing metadata

---

## Environment Variables

Override config settings via environment:

```bash
# Set pipeline root directory
export PET_PIPELINE_ROOT=/path/to/pipeline

# Disable GPU for TotalSegmentator
export TOTALSEG_DISABLE_GPU=1
```

---

## Example Configurations

### Lung Cancer Study

```yaml
modalities:
  - CT
  - PET

organs:
  - lung_upper_lobe_left
  - lung_lower_lobe_left
  - lung_upper_lobe_right
  - lung_middle_lobe_right
  - lung_lower_lobe_right
  - heart
  - aorta
  - esophagus

segmentation:
  fast: false  # Full resolution for accuracy
```

### Liver Oncology

```yaml
modalities:
  - CT
  - PET

organs:
  - liver
  - spleen
  - portal_vein_and_splenic_vein
  - gallbladder
  - pancreas
  - aorta

segmentation:
  fast: true
```

### Whole-Body Screening

```yaml
modalities:
  - CT
  - PET

organs:
  - all

segmentation:
  fast: true  # Recommended for large-scale processing
```
