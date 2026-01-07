#!/usr/bin/env python3
"""Check manufacturer and SUV scaling"""
import pydicom

ds = pydicom.dcmread("raw_download/20251226165328/PET/00001.dcm")

print("=== Manufacturer Info ===")
print(f"Manufacturer: {ds.get('Manufacturer', 'N/A')}")
print(f"Manufacturer Model: {ds.get('ManufacturerModelName', 'N/A')}")
print(f"Software Version: {ds.get('SoftwareVersions', 'N/A')}")
print(f"Station Name: {ds.get('StationName', 'N/A')}")

print("\n=== Searching for SUV-related tags ===")
for elem in ds:
    elem_str = str(elem).upper()
    if any(keyword in elem_str for keyword in ['SUV', 'SCALE', 'FACTOR', 'PHILIPS']):
        print(elem)

print("\n=== Recommendation ===")
manufacturer = ds.get('Manufacturer', '').upper()
if 'PHILIPS' in manufacturer:
    print("This is a Philips PET scanner.")
    print("Philips often uses SUV bw scale factor in private tags.")
    print("NIfTI values might already be SUV * 1000 or SUV * arbitrary factor.")
else:
    print(f"Manufacturer: {manufacturer}")
    print("Check if SUV calculation is correct for this vendor.")
