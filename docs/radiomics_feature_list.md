# Radiomics Feature List

This document describes the 107 IBSI-compliant radiomic features extracted by PyRadiomics.

## Feature Categories

| Category | Count | Description |
|----------|-------|-------------|
| First-order | 18 | Intensity distribution statistics |
| Shape | 14 | 3D morphological features |
| GLCM | 24 | Gray Level Co-occurrence Matrix |
| GLRLM | 16 | Gray Level Run Length Matrix |
| GLSZM | 16 | Gray Level Size Zone Matrix |
| GLDM | 14 | Gray Level Dependence Matrix |
| NGTDM | 5 | Neighboring Gray Tone Difference Matrix |
| **Total** | **107** | |

---

## First-order Features (18)

Statistics describing the intensity distribution within the ROI.

| Feature | Description | Unit |
|---------|-------------|------|
| Energy | Sum of squared intensities | - |
| TotalEnergy | Energy × voxel volume | mm³ |
| Entropy | Randomness of intensity distribution | bits |
| Minimum | Minimum intensity value | SUV/HU |
| 10Percentile | 10th percentile intensity | SUV/HU |
| 90Percentile | 90th percentile intensity | SUV/HU |
| Maximum | Maximum intensity value | SUV/HU |
| Mean | Average intensity | SUV/HU |
| Median | Median intensity | SUV/HU |
| InterquartileRange | IQR (75th - 25th percentile) | SUV/HU |
| Range | Maximum - Minimum | SUV/HU |
| MeanAbsoluteDeviation | Mean absolute deviation from mean | SUV/HU |
| RobustMeanAbsoluteDeviation | MAD using 10-90 percentile | SUV/HU |
| RootMeanSquared | Root mean square of intensities | SUV/HU |
| StandardDeviation | Standard deviation | SUV/HU |
| Skewness | Asymmetry of distribution | - |
| Kurtosis | Peakedness of distribution | - |
| Variance | Variance of intensities | (SUV/HU)² |
| Uniformity | Sum of squared probabilities | - |

---

## Shape Features (14)

3D morphological characteristics of the ROI.

| Feature | Description | Unit |
|---------|-------------|------|
| VoxelVolume | Volume in voxel count × voxel size | mm³ |
| MeshVolume | Volume from triangular mesh | mm³ |
| SurfaceArea | Surface area from mesh | mm² |
| SurfaceVolumeRatio | Surface area / Volume | mm⁻¹ |
| Sphericity | How spherical the shape is | - |
| Compactness1 | Volume / Surface area ratio | - |
| Compactness2 | Alternative compactness | - |
| SphericalDisproportion | 1 / Sphericity | - |
| Maximum3DDiameter | Largest pairwise distance | mm |
| Maximum2DDiameterSlice | Largest diameter in axial plane | mm |
| Maximum2DDiameterColumn | Largest diameter in coronal plane | mm |
| Maximum2DDiameterRow | Largest diameter in sagittal plane | mm |
| MajorAxisLength | Length of largest principal axis | mm |
| MinorAxisLength | Length of second principal axis | mm |
| LeastAxisLength | Length of smallest principal axis | mm |
| Elongation | Minor / Major axis ratio | - |
| Flatness | Least / Major axis ratio | - |

---

## GLCM Features (24)

Gray Level Co-occurrence Matrix captures spatial relationships between voxel pairs.

| Feature | Description |
|---------|-------------|
| Autocorrelation | Correlation of image with itself |
| JointAverage | Mean of joint probability |
| ClusterProminence | Measure of asymmetry and shape |
| ClusterShade | Skewness of GLCM |
| ClusterTendency | Grouping of similar values |
| Contrast | Local intensity variation |
| Correlation | Linear dependency of gray levels |
| DifferenceAverage | Average of difference matrix |
| DifferenceEntropy | Entropy of difference matrix |
| DifferenceVariance | Variance of difference matrix |
| JointEnergy | Sum of squared GLCM elements |
| JointEntropy | Randomness of GLCM |
| Imc1 | Information Measure of Correlation 1 |
| Imc2 | Information Measure of Correlation 2 |
| Idm | Inverse Difference Moment |
| Idmn | Normalized IDM |
| Id | Inverse Difference |
| Idn | Normalized ID |
| InverseVariance | Inverse of variance |
| MaximumProbability | Maximum GLCM element |
| SumAverage | Average of sum matrix |
| SumEntropy | Entropy of sum matrix |
| SumSquares | Variance of GLCM |
| MCC | Maximal Correlation Coefficient |

---

## GLRLM Features (16)

Gray Level Run Length Matrix describes consecutive voxels with same intensity.

| Feature | Description |
|---------|-------------|
| ShortRunEmphasis | Distribution of short runs |
| LongRunEmphasis | Distribution of long runs |
| GrayLevelNonUniformity | Variability of gray levels |
| GrayLevelNonUniformityNormalized | Normalized GLNU |
| RunLengthNonUniformity | Variability of run lengths |
| RunLengthNonUniformityNormalized | Normalized RLNU |
| RunPercentage | Fraction of runs |
| GrayLevelVariance | Variance of gray levels |
| RunVariance | Variance of run lengths |
| RunEntropy | Randomness of runs |
| LowGrayLevelRunEmphasis | Distribution of low gray level runs |
| HighGrayLevelRunEmphasis | Distribution of high gray level runs |
| ShortRunLowGrayLevelEmphasis | Short runs with low gray levels |
| ShortRunHighGrayLevelEmphasis | Short runs with high gray levels |
| LongRunLowGrayLevelEmphasis | Long runs with low gray levels |
| LongRunHighGrayLevelEmphasis | Long runs with high gray levels |

---

## GLSZM Features (16)

Gray Level Size Zone Matrix describes connected regions of same intensity.

| Feature | Description |
|---------|-------------|
| SmallAreaEmphasis | Distribution of small zones |
| LargeAreaEmphasis | Distribution of large zones |
| GrayLevelNonUniformity | Variability of gray levels |
| GrayLevelNonUniformityNormalized | Normalized GLNU |
| SizeZoneNonUniformity | Variability of zone sizes |
| SizeZoneNonUniformityNormalized | Normalized SZNU |
| ZonePercentage | Fraction of zones |
| GrayLevelVariance | Variance of gray levels |
| ZoneVariance | Variance of zone sizes |
| ZoneEntropy | Randomness of zones |
| LowGrayLevelZoneEmphasis | Distribution of low gray level zones |
| HighGrayLevelZoneEmphasis | Distribution of high gray level zones |
| SmallAreaLowGrayLevelEmphasis | Small zones with low gray levels |
| SmallAreaHighGrayLevelEmphasis | Small zones with high gray levels |
| LargeAreaLowGrayLevelEmphasis | Large zones with low gray levels |
| LargeAreaHighGrayLevelEmphasis | Large zones with high gray levels |

---

## GLDM Features (14)

Gray Level Dependence Matrix quantifies connected voxels within distance.

| Feature | Description |
|---------|-------------|
| SmallDependenceEmphasis | Distribution of small dependencies |
| LargeDependenceEmphasis | Distribution of large dependencies |
| GrayLevelNonUniformity | Variability of gray levels |
| DependenceNonUniformity | Variability of dependencies |
| DependenceNonUniformityNormalized | Normalized DNU |
| GrayLevelVariance | Variance of gray levels |
| DependenceVariance | Variance of dependencies |
| DependenceEntropy | Randomness of dependencies |
| LowGrayLevelEmphasis | Distribution of low gray levels |
| HighGrayLevelEmphasis | Distribution of high gray levels |
| SmallDependenceLowGrayLevelEmphasis | Small dependencies, low gray |
| SmallDependenceHighGrayLevelEmphasis | Small dependencies, high gray |
| LargeDependenceLowGrayLevelEmphasis | Large dependencies, low gray |
| LargeDependenceHighGrayLevelEmphasis | Large dependencies, high gray |

---

## NGTDM Features (5)

Neighboring Gray Tone Difference Matrix describes local texture patterns.

| Feature | Description |
|---------|-------------|
| Coarseness | Spatial rate of change |
| Contrast | Range of gray levels |
| Busyness | Rapid changes in intensity |
| Complexity | Primitives in image |
| Strength | Edge definition |

---

## References

- [PyRadiomics Documentation](https://pyradiomics.readthedocs.io/en/latest/features.html)
- [IBSI Reference Manual](https://theibsi.github.io/ibsi1/)
- Zwanenburg A, et al. The Image Biomarker Standardization Initiative. Radiology. 2020;295(2):328-338.
