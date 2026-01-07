#!/usr/bin/env python3
"""Check SUV parameters from DICOM"""
import pydicom
from pathlib import Path

dicom_dir = Path("raw_download/20251226165328/PET")
files = sorted(list(dicom_dir.glob("*.dcm")))

if files:
    ds = pydicom.dcmread(str(files[0]))

    print("=== DICOM SUV Parameters ===")
    print(f"Patient Weight: {ds.get('PatientWeight', 'N/A')} kg")

    if hasattr(ds, 'RadiopharmaceuticalInformationSequence'):
        rp = ds.RadiopharmaceuticalInformationSequence[0]
        total_dose = float(rp.get('RadionuclideTotalDose', 0))
        print(f"Total Dose: {total_dose} Bq")
        print(f"Total Dose: {total_dose/1e6:.1f} MBq")

    print(f"\nRescale Slope: {ds.get('RescaleSlope', 'N/A')}")
    print(f"Rescale Intercept: {ds.get('RescaleIntercept', 'N/A')}")

    if hasattr(ds, 'Units'):
        print(f"Units: {ds.Units}")
    else:
        print("Units: Not found in DICOM")

    # ピクセル値の統計
    pixel_array = ds.pixel_array
    print(f"\nPixel Array Statistics:")
    print(f"  Min: {pixel_array.min()}")
    print(f"  Max: {pixel_array.max()}")
    print(f"  Mean: {pixel_array.mean():.2f}")
    print(f"  Median: {float(pixel_array.flatten()[len(pixel_array.flatten())//2]):.2f}")

    # Rescale適用後の値
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        rescaled = pixel_array * slope + intercept
        print(f"\nRescaled Values:")
        print(f"  Min: {rescaled.min():.4f}")
        print(f"  Max: {rescaled.max():.4f}")
        print(f"  Mean: {rescaled.mean():.4f}")
