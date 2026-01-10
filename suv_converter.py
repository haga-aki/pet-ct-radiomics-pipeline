#!/usr/bin/env python
"""
PET SUV Conversion Module
=========================
Automatic vendor-specific SUV calculation

Supported vendors:
- TOSHIBA/Canon: Private tag (7065,102D) for SUVbw(X100) verification
- Siemens: Calculate from Bq/ml
- GE: Calculate from Bq/ml
- Philips: Private tag (7053,1000) SUVScaleFactor
- Generic: Calculate from standard DICOM tags

SUVbw = Activity(Bq/ml) * Weight(kg) / InjectedDose(Bq) * DecayCorrection
"""

import pydicom
import numpy as np
from datetime import datetime
from pathlib import Path


class SUVConverter:
    """Vendor-specific SUV conversion class"""

    def __init__(self, dicom_dir):
        """
        Args:
            dicom_dir: Path to PET DICOM folder
        """
        self.dicom_dir = Path(dicom_dir)
        self.ds = self._load_sample_dicom()
        self.manufacturer = self._detect_manufacturer()
        self.suv_info = self._analyze_suv_method()

    def _load_sample_dicom(self):
        """Load a sample DICOM file"""
        for f in self.dicom_dir.iterdir():
            if f.is_file() and not f.name.startswith('.'):
                try:
                    return pydicom.dcmread(str(f), stop_before_pixels=True)
                except:
                    continue
        raise ValueError(f"No valid DICOM files in {self.dicom_dir}")

    def _detect_manufacturer(self):
        """Detect manufacturer"""
        manufacturer = getattr(self.ds, 'Manufacturer', '').upper()

        if 'TOSHIBA' in manufacturer or 'CANON' in manufacturer:
            return 'TOSHIBA'
        elif 'SIEMENS' in manufacturer:
            return 'SIEMENS'
        elif 'GE' in manufacturer or 'GENERAL ELECTRIC' in manufacturer:
            return 'GE'
        elif 'PHILIPS' in manufacturer:
            return 'PHILIPS'
        else:
            return 'GENERIC'

    def _analyze_suv_method(self):
        """Analyze SUV calculation method"""
        info = {
            'manufacturer': self.manufacturer,
            'method': None,
            'scale_factor': None,
            'units': getattr(self.ds, 'Units', 'UNKNOWN'),
            'needs_calculation': False,
            'validated': False,  # True if vendor-specific method confirmed
        }

        if self.manufacturer == 'TOSHIBA':
            info.update(self._analyze_toshiba())
        elif self.manufacturer == 'PHILIPS':
            info.update(self._analyze_philips())
        elif self.manufacturer in ['SIEMENS', 'GE']:
            info.update(self._analyze_standard())
        else:
            info.update(self._analyze_standard())

        return info

    def _analyze_toshiba(self):
        """Analyze TOSHIBA/Canon SUV"""
        # Check private tag (7065, 102D)
        suv_tag = self.ds.get((0x7065, 0x102D))

        if suv_tag:
            suv_type = suv_tag.value.decode('utf-8').strip() if isinstance(suv_tag.value, bytes) else str(suv_tag.value).strip()

            if 'SUVbw' in suv_type and 'X100' in suv_type:
                return {
                    'method': 'TOSHIBA_SUVbw_X100',
                    'scale_factor': 100.0,
                    'description': f'Pixel value / 100 = SUVbw ({suv_type})',
                    'needs_calculation': False,
                    'validated': True,
                }
            elif 'SUVbw' in suv_type:
                return {
                    'method': 'TOSHIBA_SUVbw',
                    'scale_factor': 1.0,
                    'description': f'Pixel value = SUVbw ({suv_type})',
                    'needs_calculation': False,
                    'validated': True,
                }

        # Fallback: Calculate from Bq/ml
        return self._analyze_standard()

    def _analyze_philips(self):
        """Analyze Philips SUV"""
        # Private tag (7053, 1000) SUVScaleFactor
        suv_scale = self.ds.get((0x7053, 0x1000))

        if suv_scale:
            return {
                'method': 'PHILIPS_SUV_SCALE',
                'scale_factor': float(suv_scale.value),
                'description': f'Pixel value * {suv_scale.value} = SUV',
                'needs_calculation': False,
                'validated': True,
            }

        # Activity Concentration Scale Factor
        act_scale = self.ds.get((0x7053, 0x1009))
        if act_scale:
            return {
                'method': 'PHILIPS_ACTIVITY',
                'scale_factor': float(act_scale.value),
                'description': 'Need to calculate SUV from activity',
                'needs_calculation': True,
                'validated': True,
            }

        return self._analyze_standard()

    def _analyze_standard(self):
        """Standard SUV calculation from Bq/ml"""
        # Collect required information
        patient_weight = getattr(self.ds, 'PatientWeight', None)

        rp_info = None
        if hasattr(self.ds, 'RadiopharmaceuticalInformationSequence'):
            rp = self.ds.RadiopharmaceuticalInformationSequence[0]
            rp_info = {
                'total_dose': getattr(rp, 'RadionuclideTotalDose', None),
                'half_life': getattr(rp, 'RadionuclideHalfLife', None),
                'start_time': getattr(rp, 'RadiopharmaceuticalStartTime', None),
            }

        can_calculate = (patient_weight and rp_info and
                        rp_info['total_dose'] and rp_info['half_life'])

        if can_calculate:
            return {
                'method': 'STANDARD_BQML',
                'scale_factor': None,
                'description': 'Calculate SUVbw from Bq/ml using decay correction',
                'needs_calculation': True,
                'validated': True,
                'patient_weight': patient_weight,
                'radiopharmaceutical': rp_info,
            }
        else:
            return {
                'method': 'UNKNOWN',
                'scale_factor': None,
                'description': 'Cannot determine SUV calculation method',
                'needs_calculation': True,
                'validated': False,
            }

    def get_suv_scale_factor(self):
        """
        Get SUV conversion factor

        Returns:
            float: Divide pixel value by this to get SUV
                   None if manual calculation is required
        """
        if self.suv_info['method'] == 'TOSHIBA_SUVbw_X100':
            return 100.0
        elif self.suv_info['method'] == 'TOSHIBA_SUVbw':
            return 1.0
        elif self.suv_info['method'] == 'PHILIPS_SUV_SCALE':
            # Philips uses multiplication, so return inverse
            return 1.0 / self.suv_info['scale_factor']
        else:
            return None

    def calculate_suv_from_bqml(self, bqml_value, acquisition_time=None):
        """
        Calculate SUVbw from Bq/ml values

        Args:
            bqml_value: Bq/ml value (numpy array supported)
            acquisition_time: Acquisition time (HHMMSS or HHMMSS.ffffff format)

        Returns:
            SUVbw values
        """
        if not self.suv_info.get('needs_calculation', True):
            raise ValueError("This data doesn't need Bq/ml calculation")

        if self.suv_info['method'] == 'UNKNOWN':
            raise ValueError("Cannot calculate SUV: missing required DICOM tags")

        weight = self.suv_info.get('patient_weight')  # kg
        rp_info = self.suv_info.get('radiopharmaceutical', {})

        injected_dose = rp_info.get('total_dose')  # Bq
        half_life = rp_info.get('half_life')  # seconds
        start_time = rp_info.get('start_time')  # injection time

        if acquisition_time is None:
            acquisition_time = getattr(self.ds, 'AcquisitionTime',
                                      getattr(self.ds, 'SeriesTime', None))

        # Calculate time difference (seconds)
        def parse_time(t):
            t_str = str(t).split('.')[0]
            if len(t_str) >= 6:
                return int(t_str[0:2]) * 3600 + int(t_str[2:4]) * 60 + int(t_str[4:6])
            return 0

        time_diff = parse_time(acquisition_time) - parse_time(start_time)
        if time_diff < 0:
            time_diff += 24 * 3600  # Handle day crossing

        # Decay correction
        decay_factor = np.exp(-np.log(2) * time_diff / half_life)
        corrected_dose = injected_dose * decay_factor

        # SUVbw = Activity(Bq/ml) / (CorrectedDose(Bq) / Weight(g))
        # Convert weight to grams (kg * 1000)
        suv = bqml_value / (corrected_dose / (weight * 1000))

        return suv

    def convert_to_suv(self, pixel_data):
        """
        Convert pixel data to SUV

        Args:
            pixel_data: numpy array of pixel values

        Returns:
            SUV values (numpy array)
        """
        scale = self.get_suv_scale_factor()

        if scale is not None:
            return pixel_data / scale
        else:
            # Calculation from Bq/ml required
            rescale_slope = float(getattr(self.ds, 'RescaleSlope', 1))
            rescale_intercept = float(getattr(self.ds, 'RescaleIntercept', 0))

            # Apply rescale
            bqml = pixel_data * rescale_slope + rescale_intercept

            return self.calculate_suv_from_bqml(bqml)

    def print_info(self):
        """Display SUV conversion information"""
        print(f"\n{'='*60}")
        print(f"PET SUV Conversion Info")
        print(f"{'='*60}")
        print(f"Manufacturer: {self.manufacturer}")
        print(f"Model: {getattr(self.ds, 'ManufacturerModelName', 'N/A')}")
        print(f"Units in DICOM: {self.suv_info['units']}")
        print(f"Method: {self.suv_info['method']}")
        print(f"Description: {self.suv_info['description']}")
        print(f"Validated: {self.suv_info.get('validated', False)}")

        scale = self.get_suv_scale_factor()
        if scale:
            print(f"Scale Factor: Pixel / {scale} = SUV")
        else:
            print(f"Calculation: Requires decay correction from Bq/ml")
        print()


def detect_suv_scale(dicom_dir):
    """
    Detect SUV scale factor from DICOM folder

    Args:
        dicom_dir: Path to PET DICOM folder

    Returns:
        tuple: (scale_factor, method_description)
               scale_factor is None if calculation from Bq/ml is required
    """
    converter = SUVConverter(dicom_dir)
    converter.print_info()
    return converter.get_suv_scale_factor(), converter.suv_info['method']


if __name__ == "__main__":
    # Test
    import sys

    if len(sys.argv) > 1:
        dicom_dir = sys.argv[1]
    else:
        print("Usage: python suv_converter.py <dicom_directory>")
        print("Example: python suv_converter.py ./raw_download/patient_001/PET")
        sys.exit(1)

    converter = SUVConverter(dicom_dir)
    converter.print_info()

    # Test conversion
    scale = converter.get_suv_scale_factor()
    if scale:
        test_value = 500  # Sample pixel value
        suv = test_value / scale
        print(f"Test: Pixel value {test_value} -> SUV {suv:.3f}")
