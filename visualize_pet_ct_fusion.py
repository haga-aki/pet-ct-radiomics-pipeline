#!/usr/bin/env python3
"""
PET-CT Fusion画像可視化スクリプト
PETとCTのfusion画像にセグメンテーションマスクをオーバーレイ
SUV最大値のスライスを含む
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk

def normalize_image(img_data, percentile_low=1, percentile_high=99):
    """画像をパーセンタイルベースで正規化"""
    vmin = np.percentile(img_data, percentile_low)
    vmax = np.percentile(img_data, percentile_high)
    img_norm = np.clip(img_data, vmin, vmax)
    if vmax > vmin:
        return (img_norm - vmin) / (vmax - vmin)
    return img_data

def calculate_suv_from_dicom(case_id):
    """DICOMからSUVパラメータを取得"""
    import pydicom
    import glob
    from datetime import datetime

    base_dir = Path(".")

    # ID mappingから元のフォルダ名を取得
    id_map_file = base_dir / "id_mapping.csv"
    original_folder = None

    if id_map_file.exists():
        import csv
        with open(id_map_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # ヘッダースキップ
            for row in reader:
                if len(row) >= 2 and row[1] == case_id:
                    original_folder = row[0]
                    break

    if not original_folder:
        print(f"  Warning: Could not find original folder for {case_id}")
        return None

    dicom_dir = base_dir / "raw_download" / original_folder
    pet_dicom_dir = dicom_dir / "PET"

    if not pet_dicom_dir.exists():
        print(f"  Warning: PET DICOM not found at {pet_dicom_dir}")
        return None

    try:
        files = sorted(list(pet_dicom_dir.glob("*.dcm")))
        if not files:
            return None

        ds = pydicom.dcmread(str(files[0]))

        # 必要なパラメータを取得
        weight_kg = float(ds.get('PatientWeight', 0))
        if weight_kg == 0:
            print("  Warning: Patient weight not found in DICOM")
            return None

        weight_g = weight_kg * 1000

        if hasattr(ds, 'RadiopharmaceuticalInformationSequence'):
            rp = ds.RadiopharmaceuticalInformationSequence[0]
            injected_dose = float(rp.get('RadionuclideTotalDose', 0))

            if injected_dose == 0:
                print("  Warning: Injected dose not found")
                return None

            # 減衰補正（オプション）
            # 簡易版: 減衰は考慮せず投与量をそのまま使用
            # 本格的には半減期と時間差を考慮

            return {
                'weight_g': weight_g,
                'total_dose_bq': injected_dose,
                'weight_kg': weight_kg
            }
    except Exception as e:
        print(f"  Error reading DICOM for SUV: {e}")
        return None

    return None

def resample_pet_to_ct(pet_path, ct_path, suv_params=None):
    """PETをCT空間にリサンプリング（SUV変換含む）"""
    pet_img = sitk.ReadImage(str(pet_path))
    ct_img = sitk.ReadImage(str(ct_path))

    # SUV変換
    if suv_params:
        # PET画像をnumpy配列に変換
        pet_array = sitk.GetArrayFromImage(pet_img)

        # TOSHIBA PET: NIfTI値は SUVbw(X100) = SUV * 100
        # 正しいSUV値に変換
        suv_array = pet_array / 100.0

        # SUV画像を作成
        pet_img = sitk.GetImageFromArray(suv_array)
        pet_img.CopyInformation(sitk.ReadImage(str(pet_path)))

        print(f"  SUV conversion: TOSHIBA SUVbw(X100) format detected")
        print(f"  Converted NIfTI max {pet_array.max():.1f} → SUV max {suv_array.max():.2f}")

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    pet_resampled = resampler.Execute(pet_img)
    return sitk.GetArrayFromImage(pet_resampled).transpose(2, 1, 0), sitk.GetArrayFromImage(ct_img).transpose(2, 1, 0)

def create_pet_ct_fusion(case_id="ILD_002"):
    """PET-CT Fusion画像の作成"""

    base_dir = Path(".")
    nifti_dir = base_dir / "nifti_images"
    seg_dir = base_dir / "segmentations"
    output_dir = base_dir / "visualizations" / case_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ファイルパス
    ct_path = nifti_dir / f"{case_id}_CT.nii.gz"
    pet_path = nifti_dir / f"{case_id}_PET.nii.gz"
    ct_seg_dir = seg_dir / f"{case_id}_CT"

    if not all([ct_path.exists(), pet_path.exists(), ct_seg_dir.exists()]):
        print("Error: Required files not found")
        return

    print(f"Loading and resampling images for {case_id}...")

    # SUVパラメータ取得
    suv_params = calculate_suv_from_dicom(case_id)
    if suv_params:
        print(f"  SUV parameters loaded from DICOM")
    else:
        print(f"  Warning: Could not load SUV parameters, using raw PET values")

    # CT画像読み込み
    ct_img = nib.load(str(ct_path))
    ct_data = ct_img.get_fdata()

    # PET画像読み込みとCT空間へのリサンプリング（SUV変換含む）
    print("  Resampling PET to CT space with SUV conversion...")
    pet_resampled, ct_array = resample_pet_to_ct(pet_path, ct_path, suv_params)

    print(f"  CT shape: {ct_data.shape}")
    print(f"  PET resampled shape: {pet_resampled.shape}")

    # マスク読み込み（肺野領域の特定に使用）
    print("Loading lung masks to identify lung region...")
    lobes = ['lung_upper_lobe_left', 'lung_lower_lobe_left',
             'lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right']

    ct_mask = np.zeros(ct_data.shape, dtype=np.uint8)
    for i, lobe in enumerate(lobes, start=1):
        lobe_path = ct_seg_dir / f"{lobe}.nii.gz"
        if lobe_path.exists():
            lobe_data = nib.load(str(lobe_path)).get_fdata()
            ct_mask[lobe_data > 0] = i

    # 肺野が存在するスライス範囲を特定
    lung_slices = np.where(np.sum(ct_mask, axis=(0, 1)) > 0)[0]
    if len(lung_slices) == 0:
        print("Error: No lung region found in masks")
        return

    lung_start = lung_slices[0]
    lung_end = lung_slices[-1]
    print(f"  Lung region: slices {lung_start} to {lung_end}")

    # SUV値の統計（肺野領域のみ）
    # 肺マスク内のPET値のみを対象にする
    pet_in_lung = pet_resampled * (ct_mask > 0)
    pet_max = np.max(pet_in_lung)
    pet_mean = np.mean(pet_in_lung[pet_in_lung > 0])
    print(f"  PET SUV max (in lung region): {pet_max:.2f}")
    print(f"  PET SUV mean (>0, in lung region): {pet_mean:.2f}")

    # SUV最大値の位置を特定（肺マスク内）
    max_suv_pos = np.unravel_index(np.argmax(pet_in_lung), pet_in_lung.shape)
    max_suv_slice_axial = max_suv_pos[2]
    max_suv_slice_coronal = max_suv_pos[1]
    print(f"  Max SUV position: x={max_suv_pos[0]}, y={max_suv_pos[1]}, z={max_suv_pos[2]}")
    max_suv_slice = max_suv_slice_axial
    print(f"  Slice with max SUV uptake (in lung): {max_suv_slice}/{ct_data.shape[2]}")

    # 左右の肺を分離して統計を取得
    # 左肺: lung_upper_lobe_left (1), lung_lower_lobe_left (2)
    # 右肺: lung_upper_lobe_right (3), lung_middle_lobe_right (4), lung_lower_lobe_right (5)
    left_lung_mask = np.isin(ct_mask, [1, 2])
    right_lung_mask = np.isin(ct_mask, [3, 4, 5])

    # 左右それぞれのSUV統計
    pet_left_lung = pet_resampled[left_lung_mask]
    pet_right_lung = pet_resampled[right_lung_mask]

    left_suv_max = np.max(pet_left_lung) if len(pet_left_lung) > 0 else 0
    right_suv_max = np.max(pet_right_lung) if len(pet_right_lung) > 0 else 0

    # 最大SUVが低い方（腫瘤がない方）の肺の統計
    if left_suv_max < right_suv_max:
        contralateral_lung_name = "Left"
        contralateral_lung_values = pet_left_lung[pet_left_lung > 0]
    else:
        contralateral_lung_name = "Right"
        contralateral_lung_values = pet_right_lung[pet_right_lung > 0]

    if len(contralateral_lung_values) > 0:
        contra_suv_mean = np.mean(contralateral_lung_values)
        contra_suv_median = np.median(contralateral_lung_values)
        contra_suv_std = np.std(contralateral_lung_values)
        contra_suv_max = np.max(contralateral_lung_values)
        print(f"\n  Contralateral lung ({contralateral_lung_name}) statistics:")
        print(f"    SUV mean: {contra_suv_mean:.3f}")
        print(f"    SUV median: {contra_suv_median:.3f}")
        print(f"    SUV std: {contra_suv_std:.3f}")
        print(f"    SUV max: {contra_suv_max:.3f}")
    else:
        contra_suv_mean = contra_suv_median = contra_suv_std = contra_suv_max = 0
        print(f"\n  Warning: No contralateral lung data")

    # 可視化するスライスを選択（肺野領域内のみ）
    middle_slice = (lung_start + lung_end) // 2
    slices_to_visualize = [
        (max_suv_slice, f"Max SUV Uptake (Slice {max_suv_slice})"),
        (middle_slice, f"Middle (Slice {middle_slice})")
    ]

    for slice_idx, slice_title in slices_to_visualize:
        print(f"\nCreating fusion visualization for {slice_title}...")

        # スライス抽出
        ct_slice = ct_data[:, :, slice_idx].T
        pet_slice = pet_resampled[:, :, slice_idx].T
        mask_slice = ct_mask[:, :, slice_idx].T

        # CT画像の正規化（肺野用ウィンドウレベル）
        ct_slice_norm = np.clip(ct_slice, -1000, 500)
        ct_slice_norm = normalize_image(ct_slice_norm)

        # PET画像の正規化
        pet_slice_norm = normalize_image(pet_slice)

        # 図の作成（2行3列）
        fig = plt.figure(figsize=(20, 12))

        # 1行目: CT, PET, Fusion
        # CT画像
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(ct_slice_norm, cmap='gray', origin='lower')
        ax1.set_title('CT Image', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # PET画像（CT空間にリサンプリング済み）
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(pet_slice_norm, cmap='hot', origin='lower')
        ax2.set_title('PET (Resampled to CT)', fontsize=14, fontweight='bold')
        ax2.axis('off')

        # PET-CT Fusion
        ax3 = plt.subplot(2, 3, 3)
        ax3.imshow(ct_slice_norm, cmap='gray', origin='lower', alpha=1.0)
        pet_overlay = np.ma.masked_where(pet_slice_norm < 0.1, pet_slice_norm)
        ax3.imshow(pet_overlay, cmap='hot', origin='lower', alpha=0.6)
        ax3.set_title('PET-CT Fusion', fontsize=14, fontweight='bold')
        ax3.axis('off')

        # 2行目: CT + Mask, PET + Mask, Fusion + Mask
        # マスクの輪郭を強調表示するために、境界線を作成
        from scipy import ndimage

        # マスクの境界を検出
        mask_binary = (mask_slice > 0).astype(np.uint8)
        mask_edges = mask_binary - ndimage.binary_erosion(mask_binary, structure=np.ones((3,3)))
        mask_edges_overlay = np.ma.masked_where(mask_edges == 0, mask_edges)

        # マスク領域全体（塗りつぶし用）
        mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)

        # CT + マスク: マスクを緑-マゼンタ系の色で表示
        ax4 = plt.subplot(2, 3, 4)
        ax4.imshow(ct_slice_norm, cmap='gray', origin='lower')
        ax4.imshow(mask_overlay, cmap='spring', alpha=0.3, origin='lower', vmin=0, vmax=10)
        ax4.imshow(mask_edges_overlay, cmap='spring', alpha=0.9, origin='lower')
        ax4.set_title('CT + Lung Lobes Mask', fontsize=14, fontweight='bold')
        ax4.axis('off')

        # PET + マスク: マスクを青-シアン系の色で表示（PETのhotカラーマップと対比）
        ax5 = plt.subplot(2, 3, 5)
        ax5.imshow(pet_slice_norm, cmap='hot', origin='lower')
        ax5.imshow(mask_overlay, cmap='cool', alpha=0.25, origin='lower', vmin=0, vmax=10)
        ax5.imshow(mask_edges_overlay, cmap='cool', alpha=0.8, origin='lower')
        ax5.set_title('PET + Lung Lobes Mask', fontsize=14, fontweight='bold')
        ax5.axis('off')

        # Fusion + マスク: マスクをシアン系の輪郭で表示（PETの赤と対比）
        ax6 = plt.subplot(2, 3, 6)
        ax6.imshow(ct_slice_norm, cmap='gray', origin='lower', alpha=1.0)
        ax6.imshow(pet_overlay, cmap='hot', origin='lower', alpha=0.6)
        ax6.imshow(mask_overlay, cmap='winter', alpha=0.2, origin='lower', vmin=0, vmax=10)
        ax6.imshow(mask_edges_overlay, cmap='winter', alpha=0.7, origin='lower')
        ax6.set_title('PET-CT Fusion + Lung Lobes Mask', fontsize=14, fontweight='bold')
        ax6.axis('off')

        # 全体タイトル（統計情報を含む）
        stats_text = f'Contralateral ({contralateral_lung_name}) Lung: Mean={contra_suv_mean:.3f}, Median={contra_suv_median:.3f}, Max={contra_suv_max:.3f}'
        fig.suptitle(f'{case_id}: {slice_title} (Axial) - Whole Lung SUV max: {pet_max:.2f}\n{stats_text}',
                     fontsize=13, fontweight='bold', y=0.99)

        plt.tight_layout()

        # ファイル保存
        output_path = output_dir / f"fusion_axial_slice{slice_idx}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()

    # Coronal view可視化
    print("\nCreating coronal view visualizations...")

    # 肺領域のY方向の範囲を特定
    lung_slices_coronal = np.where(np.sum(ct_mask, axis=(0, 2)) > 0)[0]
    lung_start_coronal = lung_slices_coronal[0]
    lung_end_coronal = lung_slices_coronal[-1]
    middle_slice_coronal = (lung_start_coronal + lung_end_coronal) // 2

    print(f"  Lung region (coronal): slices {lung_start_coronal} to {lung_end_coronal}")
    print(f"  Max SUV coronal slice: {max_suv_slice_coronal}")

    # Coronalの代表スライスを可視化
    coronal_slices_to_visualize = [
        (max_suv_slice_coronal, f"Max SUV Uptake (Coronal {max_suv_slice_coronal})"),
        (middle_slice_coronal, f"Middle (Coronal {middle_slice_coronal})")
    ]

    for slice_idx, slice_title in coronal_slices_to_visualize:
        print(f"\nCreating coronal fusion visualization for {slice_title}...")

        # Coronal スライス抽出 (Y軸方向) - 肺野領域のみ
        ct_slice_full = ct_data[:, slice_idx, :].T
        pet_slice_full = pet_resampled[:, slice_idx, :].T
        mask_slice_full = ct_mask[:, slice_idx, :].T

        # 肺野領域でクロップ（Z軸方向とX軸方向）
        # Z軸方向（上下）のクロップ
        ct_slice_z = ct_slice_full[lung_start:lung_end+1, :]
        pet_slice_z = pet_slice_full[lung_start:lung_end+1, :]
        mask_slice_z = mask_slice_full[lung_start:lung_end+1, :]

        # X軸方向（左右）の肺領域を特定
        lung_cols = np.where(np.sum(mask_slice_z, axis=0) > 0)[0]
        if len(lung_cols) > 0:
            lung_left = max(0, lung_cols[0] - 20)  # 左右に20ピクセルのマージン
            lung_right = min(mask_slice_z.shape[1], lung_cols[-1] + 20)

            # X軸方向もクロップ
            ct_slice = ct_slice_z[:, lung_left:lung_right]
            pet_slice = pet_slice_z[:, lung_left:lung_right]
            mask_slice = mask_slice_z[:, lung_left:lung_right]
        else:
            ct_slice = ct_slice_z
            pet_slice = pet_slice_z
            mask_slice = mask_slice_z

        # CT画像の正規化
        ct_slice_norm = np.clip(ct_slice, -1000, 500)
        ct_slice_norm = normalize_image(ct_slice_norm)

        # PET画像の正規化
        pet_slice_norm = normalize_image(pet_slice)

        # 冠状断のアスペクト比を計算
        # 冠状断: 縦=Z軸スペーシング(5.0mm)、横=X軸スペーシング(1.073mm)
        # aspect = Z_spacing / X_spacing = 5.0 / 1.073 ≈ 4.66
        height, width = ct_slice.shape
        pixel_aspect_ratio = height / width  # ピクセル数の比

        # CT画像のスペーシング情報を取得
        ct_spacing = ct_img.header.get_zooms()  # (X, Y, Z) in mm
        physical_aspect = ct_spacing[2] / ct_spacing[0]  # Z / X

        print(f"    Coronal slice shape: {height} x {width}, pixel aspect: {pixel_aspect_ratio:.3f}")
        print(f"    CT spacing: {ct_spacing}, physical aspect (Z/X): {physical_aspect:.3f}")

        # 図のサイズを調整
        # 横長の図にして、高さは物理的なアスペクト比に基づいて設定
        fig_width = 18
        fig_height = 8  # 高さを増やして物理的なアスペクト比を表現

        # 図の作成
        fig, axes = plt.subplots(2, 3, figsize=(fig_width, fig_height))

        # 1行目: CT, PET, Fusion
        # aspect='auto'ではなく、物理的なアスペクト比を指定
        axes[0, 0].imshow(ct_slice_norm, cmap='gray', origin='lower', aspect=physical_aspect)
        axes[0, 0].set_title('CT (Coronal)', fontsize=10, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(pet_slice_norm, cmap='hot', origin='lower', aspect=physical_aspect)
        axes[0, 1].set_title('PET (Coronal)', fontsize=10, fontweight='bold')
        axes[0, 1].axis('off')

        pet_overlay = np.ma.masked_where(pet_slice_norm < 0.1, pet_slice_norm)
        axes[0, 2].imshow(ct_slice_norm, cmap='gray', origin='lower', alpha=1.0, aspect=physical_aspect)
        axes[0, 2].imshow(pet_overlay, cmap='hot', origin='lower', alpha=0.6, aspect=physical_aspect)
        axes[0, 2].set_title('Fusion (Coronal)', fontsize=10, fontweight='bold')
        axes[0, 2].axis('off')

        # 2行目: CT + Mask, PET + Mask, Fusion + Mask
        # マスクの輪郭を強調表示するために、境界線を作成
        from scipy import ndimage

        # マスクの境界を検出
        mask_binary = (mask_slice > 0).astype(np.uint8)
        mask_edges = mask_binary - ndimage.binary_erosion(mask_binary, structure=np.ones((3,3)))
        mask_edges_overlay = np.ma.masked_where(mask_edges == 0, mask_edges)

        # マスク領域全体（塗りつぶし用）
        mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)

        # CT + Mask: マスクを緑色の輪郭で表示
        axes[1, 0].imshow(ct_slice_norm, cmap='gray', origin='lower', aspect=physical_aspect)
        axes[1, 0].imshow(mask_overlay, cmap='spring', alpha=0.3, origin='lower', vmin=0, vmax=10, aspect=physical_aspect)
        axes[1, 0].imshow(mask_edges_overlay, cmap='spring', alpha=0.9, origin='lower', aspect=physical_aspect)
        axes[1, 0].set_title('CT + Lung Mask (Coronal)', fontsize=10, fontweight='bold')
        axes[1, 0].axis('off')

        # PET + Mask: マスクを青-緑系の色で表示（PETのhotカラーマップと対比）
        axes[1, 1].imshow(pet_slice_norm, cmap='hot', origin='lower', aspect=physical_aspect)
        axes[1, 1].imshow(mask_overlay, cmap='cool', alpha=0.25, origin='lower', vmin=0, vmax=10, aspect=physical_aspect)
        axes[1, 1].imshow(mask_edges_overlay, cmap='cool', alpha=0.8, origin='lower', aspect=physical_aspect)
        axes[1, 1].set_title('PET + Lung Mask (Coronal)', fontsize=10, fontweight='bold')
        axes[1, 1].axis('off')

        # Fusion + Mask: マスクをシアン系の輪郭で表示（PETの赤と対比）
        axes[1, 2].imshow(ct_slice_norm, cmap='gray', origin='lower', alpha=1.0, aspect=physical_aspect)
        axes[1, 2].imshow(pet_overlay, cmap='hot', origin='lower', alpha=0.6, aspect=physical_aspect)
        axes[1, 2].imshow(mask_overlay, cmap='winter', alpha=0.2, origin='lower', vmin=0, vmax=10, aspect=physical_aspect)
        axes[1, 2].imshow(mask_edges_overlay, cmap='winter', alpha=0.7, origin='lower', aspect=physical_aspect)
        axes[1, 2].set_title('Fusion + Lung Mask (Coronal)', fontsize=10, fontweight='bold')
        axes[1, 2].axis('off')

        # 全体タイトル（統計情報を含む）
        stats_text = f'Contralateral ({contralateral_lung_name}) Lung: Mean={contra_suv_mean:.3f}, Median={contra_suv_median:.3f}, Max={contra_suv_max:.3f}'
        fig.suptitle(f'{case_id}: {slice_title} (Coronal) - Whole Lung SUV max: {pet_max:.2f}\n{stats_text}',
                     fontsize=11, fontweight='bold', y=0.98)

        # ファイル保存
        output_path = output_dir / f"fusion_coronal_slice{slice_idx}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.2)
        print(f"  Saved: {output_path}")
        plt.close()

    print(f"\n✓ Fusion visualization complete! Check {output_dir}/ for PNG files.")

if __name__ == "__main__":
    import sys

    # コマンドライン引数で症例IDを指定可能
    if len(sys.argv) > 1:
        case_id = sys.argv[1]
        create_pet_ct_fusion(case_id)
    else:
        # 引数がない場合は、PETデータがある全症例を処理
        base_dir = Path(".")
        nifti_dir = base_dir / "nifti_images"

        # PET画像がある症例を検索
        pet_files = sorted(list(nifti_dir.glob("*_PET.nii.gz")))

        if not pet_files:
            print("No PET images found. Exiting.")
            sys.exit(1)

        print(f"Found {len(pet_files)} cases with PET data.")

        for pet_file in pet_files:
            case_id = pet_file.name.replace("_PET.nii.gz", "")
            print(f"\n{'='*60}")
            print(f"Processing case: {case_id}")
            print(f"{'='*60}")

            try:
                create_pet_ct_fusion(case_id)
            except Exception as e:
                print(f"Error processing {case_id}: {e}")
                continue

        print(f"\n{'='*60}")
        print("All cases processed!")
        print(f"{'='*60}")
