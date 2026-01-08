#!/usr/bin/env python
"""
肺セグメンテーションの確認用画像を生成
CT画像に肺葉のセグメンテーションをオーバーレイ
複数症例対応版
"""
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

# Base directory - defaults to script location
BASE_DIR = Path(os.environ.get("PET_PIPELINE_ROOT", Path(__file__).parent))
NIFTI_DIR = BASE_DIR / "nifti_images"
SEG_DIR = BASE_DIR / "segmentations"
OUTPUT_DIR = BASE_DIR / "analysis_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# 肺葉の設定
LUNG_LOBES = {
    'lung_upper_lobe_left': ('Left Upper Lobe', '#3498db'),
    'lung_lower_lobe_left': ('Left Lower Lobe', '#2980b9'),
    'lung_upper_lobe_right': ('Right Upper Lobe', '#e74c3c'),
    'lung_middle_lobe_right': ('Right Middle Lobe', '#f39c12'),
    'lung_lower_lobe_right': ('Right Lower Lobe', '#d35400'),
}


def get_patient_ids():
    """segmentationsフォルダから患者IDリストを取得"""
    if not SEG_DIR.exists():
        return []
    return [p.name for p in SEG_DIR.iterdir() if p.is_dir()]


def visualize_patient(patient_id):
    """1患者の肺セグメンテーション画像を生成"""
    print(f"\n{'='*60}")
    print(f"Processing: {patient_id}")
    print(f"{'='*60}")

    ct_path = NIFTI_DIR / f"{patient_id}_CT.nii.gz"
    seg_dir = SEG_DIR / patient_id

    if not ct_path.exists():
        print(f"  CT not found: {ct_path}")
        return

    print("Loading images...")
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    print(f"CT shape: {ct_data.shape}")

    # 全肺葉のマスクを読み込み
    lobe_data = {}
    combined_mask = np.zeros_like(ct_data)

    for i, (lobe_name, (label, color)) in enumerate(LUNG_LOBES.items(), 1):
        mask_path = seg_dir / f"{lobe_name}.nii.gz"
        if mask_path.exists():
            mask = nib.load(mask_path).get_fdata()
            lobe_data[lobe_name] = {'data': mask, 'label': label, 'color': color, 'id': i}
            combined_mask[mask > 0] = i
            voxel_count = np.sum(mask > 0)
            print(f"  {label}: {voxel_count:,} voxels")

    if not lobe_data:
        print("  No lung lobe segmentations found")
        return

    # マスクが存在するスライス範囲
    seg_z = np.where(np.any(combined_mask > 0, axis=(0, 1)))[0]
    z_min, z_max = seg_z.min(), seg_z.max()
    print(f"Segmentation Z range: {z_min} to {z_max}")

    # 凡例要素
    legend_elements = [plt.Line2D([0], [0], color=info['color'], linewidth=4, label=info['label'])
                       for info in lobe_data.values()]

    # ========================================
    # Figure 1: アキシャル断面（複数スライス）
    # ========================================
    print("Generating axial view...")

    n_slices = 5
    slice_indices = [z_min + int((z_max - z_min) * i / (n_slices - 1)) for i in range(n_slices)]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for col, z_idx in enumerate(slice_indices):
        ct_slice = ct_data[:, :, z_idx].T

        # 上段: CT画像のみ
        ax = axes[0, col]
        ax.imshow(ct_slice, cmap='gray', origin='lower', vmin=-1000, vmax=400)
        ax.set_title(f'Slice {z_idx}', fontsize=11)
        ax.axis('off')

        # 下段: CT + セグメンテーション
        ax = axes[1, col]
        ax.imshow(ct_slice, cmap='gray', origin='lower', vmin=-1000, vmax=400)

        for lobe_name, info in lobe_data.items():
            mask_slice = info['data'][:, :, z_idx].T
            if np.any(mask_slice > 0):
                masked = np.ma.masked_where(mask_slice == 0, mask_slice)
                ax.imshow(masked, cmap=ListedColormap(['none', info['color']]),
                         vmin=0, vmax=1, alpha=0.3, origin='lower')
                ax.contour(mask_slice, levels=[0.5], colors=[info['color']], linewidths=1.5)

        ax.axis('off')

    axes[0, 0].set_ylabel('CT Only', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('CT + Segmentation', fontsize=12, fontweight='bold')

    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, 0.02))

    plt.suptitle(f'Lung Lobe Segmentation - {patient_id} - Axial View', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    output_path = OUTPUT_DIR / f'{patient_id}_lung_segmentation_axial.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

    # ========================================
    # Figure 2: 3方向断面
    # ========================================
    print("Generating multi-view...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    z_mid = (z_min + z_max) // 2
    y_mid = ct_data.shape[1] // 2
    x_mid = ct_data.shape[0] // 2

    views = [
        ('Axial', ct_data[:, :, z_mid].T,
         {k: v['data'][:, :, z_mid].T for k, v in lobe_data.items()}, 'equal'),
        ('Coronal', ct_data[:, y_mid, :].T,
         {k: v['data'][:, y_mid, :].T for k, v in lobe_data.items()}, 'auto'),
        ('Sagittal', ct_data[x_mid, :, :].T,
         {k: v['data'][x_mid, :, :].T for k, v in lobe_data.items()}, 'auto'),
    ]

    for col, (title, ct_slice, mask_slices, aspect) in enumerate(views):
        ax = axes[0, col]
        ax.imshow(ct_slice, cmap='gray', origin='lower', vmin=-1000, vmax=400, aspect=aspect)
        ax.set_title(f'{title} View', fontsize=12, fontweight='bold')
        ax.axis('off')

        ax = axes[1, col]
        ax.imshow(ct_slice, cmap='gray', origin='lower', vmin=-1000, vmax=400, aspect=aspect)

        for lobe_name, mask_slice in mask_slices.items():
            if np.any(mask_slice > 0):
                info = lobe_data[lobe_name]
                masked = np.ma.masked_where(mask_slice == 0, mask_slice)
                ax.imshow(masked, cmap=ListedColormap(['none', info['color']]),
                         vmin=0, vmax=1, alpha=0.3, origin='lower', aspect=aspect)
                ax.contour(mask_slice, levels=[0.5], colors=[info['color']], linewidths=1.5)
        ax.axis('off')

    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, 0.02))

    plt.suptitle(f'Lung Lobe Segmentation - {patient_id} - Multi-View', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    output_path = OUTPUT_DIR / f'{patient_id}_lung_segmentation_multiview.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

    # ========================================
    # Figure 3: ボリュームサマリー
    # ========================================
    print("Generating volume summary...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    lobe_names = [info['label'] for info in lobe_data.values()]
    lobe_volumes = [np.sum(info['data'] > 0) * np.prod(ct_img.header.get_zooms()) / 1000 for info in lobe_data.values()]
    colors_list = [info['color'] for info in lobe_data.values()]

    bars = ax.barh(lobe_names, lobe_volumes, color=colors_list, edgecolor='black')
    ax.set_xlabel('Volume (cm³)', fontsize=12)
    ax.set_title('Lung Lobe Volumes', fontsize=14, fontweight='bold')
    for bar, vol in zip(bars, lobe_volumes):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                f'{vol:.0f}', va='center', fontsize=11)
    ax.set_xlim(0, max(lobe_volumes) * 1.2 if lobe_volumes else 1)

    ax = axes[1]
    lung_mask = combined_mask > 0
    ct_lung = ct_data.copy()
    ct_lung[~lung_mask] = -1000
    mip_coronal = np.max(ct_lung, axis=1).T
    ax.imshow(mip_coronal, cmap='gray', origin='lower', vmin=-1000, vmax=400, aspect='auto')
    ax.set_title('Maximum Intensity Projection (Coronal)\nLung Region Only', fontsize=12, fontweight='bold')
    ax.axis('off')

    plt.suptitle(f'{patient_id} - Lung Volume Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = OUTPUT_DIR / f'{patient_id}_lung_segmentation_volume.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

    # ========================================
    # Figure 4: 各肺葉の個別表示
    # ========================================
    print("Generating individual lobe views...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    z_mid = (z_min + z_max) // 2
    ct_slice = ct_data[:, :, z_mid].T

    for idx, (lobe_name, info) in enumerate(lobe_data.items()):
        if idx >= 5:
            break
        ax = axes[idx]
        mask_slice = info['data'][:, :, z_mid].T

        ax.imshow(ct_slice, cmap='gray', origin='lower', vmin=-1000, vmax=400)

        if np.any(mask_slice > 0):
            masked = np.ma.masked_where(mask_slice == 0, mask_slice)
            ax.imshow(masked, cmap=ListedColormap(['none', info['color']]),
                     vmin=0, vmax=1, alpha=0.4, origin='lower')
            ax.contour(mask_slice, levels=[0.5], colors=[info['color']], linewidths=2)

        vol = np.sum(info['data'] > 0) * np.prod(ct_img.header.get_zooms()) / 1000
        ax.set_title(f"{info['label']}\nVolume: {vol:.0f} cm³", fontsize=11, fontweight='bold', color=info['color'])
        ax.axis('off')

    ax = axes[5]
    ax.imshow(ct_slice, cmap='gray', origin='lower', vmin=-1000, vmax=400)
    for lobe_name, info in lobe_data.items():
        mask_slice = info['data'][:, :, z_mid].T
        if np.any(mask_slice > 0):
            ax.contour(mask_slice, levels=[0.5], colors=[info['color']], linewidths=2)
    ax.set_title('All Lobes Combined', fontsize=11, fontweight='bold')
    ax.axis('off')

    plt.suptitle(f'{patient_id} - Individual Lung Lobes (Slice {z_mid})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = OUTPUT_DIR / f'{patient_id}_lung_segmentation_individual.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("Lung Segmentation Visualization (複数症例対応)")
    print("=" * 70)

    patient_ids = get_patient_ids()
    if not patient_ids:
        print("No patients found in segmentations folder")
        return

    print(f"\n対象患者数: {len(patient_ids)}")
    for pid in patient_ids:
        print(f"  - {pid}")

    for patient_id in sorted(patient_ids):
        visualize_patient(patient_id)

    print("\nDone!")


if __name__ == "__main__":
    main()
