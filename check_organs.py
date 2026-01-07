#!/usr/bin/env python3
"""Check if organs in config.yaml are valid TotalSegmentator organs"""
from totalsegmentator.map_to_binary import class_map

organs_in_config = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
    "heart",
    "aorta",
    "pulmonary_artery",
    "trachea",
    "esophagus"
]

available_organs = list(class_map["total"].values())

print("Checking roi_subset organs from config.yaml:\n")
for org in organs_in_config:
    status = "✓ OK" if org in available_organs else "✗ NOT FOUND"
    print(f"  {org:30s} {status}")

print(f"\nTotal available organs: {len(available_organs)}")
print("\nSearching for similar lung/chest organs:")
chest_organs = [o for o in available_organs if any(keyword in o.lower() for keyword in ['lung', 'heart', 'aorta', 'pulmonary', 'trachea', 'esophagus'])]
for org in sorted(chest_organs):
    print(f"  - {org}")
