#!/Users/akira/miniforge3/envs/med_ai/bin/python
"""
CT、PET、セグメンテーションを並べて表示して位置合わせを確認
複数症例対応版
"""
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

BASE_DIR = Path("/Users/akira/Local/medical_ai/ycu_project")
NIFTI_DIR = BASE_DIR / "nifti_images"
SEG_DIR = BASE_DIR / "segmentations"
OUTPUT_DIR = BASE_DIR / "analysis_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# 肺葉の色設定
LUNG_LOBES = {
    'lung_upper_lobe_left': ('Left Upper', '#3498db'),
    'lung_lower_lobe_left': ('Left Lower', '#2980b9'),
    'lung_upper_lobe_right': ('Right Upper', '#e74c3c'),
    'lung_middle_lobe_right': ('Right Middle', '#c0392b'),
    'lung_lower_lobe_right': ('Right Lower', '#d35400'),
}


def get_patient_ids():
    """segmentationsフォルダから患者IDリストを取得"""
    if not SEG_DIR.exists():
        return []
    return [p.name for p in SEG_DIR.iterdir() if p.is_dir()]


def visualize_patient(patient_id):
    """1患者のCT/PET位置合わせ確認画像を生成"""
    print(f"\n{'='*60}")
    print(f"Processing: {patient_id}")
    print(f"{'='*60}")

    ct_path = NIFTI_DIR / f"{patient_id}_CT.nii.gz"
    # SUV変換済みの位置合わせPETを優先
    pet_reg_path = NIFTI_DIR / f"{patient_id}_PET_SUV_registered.nii.gz"
    if not pet_reg_path.exists():
        pet_reg_path = NIFTI_DIR / f"{patient_id}_PET_registered.nii.gz"
    seg_dir = SEG_DIR / patient_id

    if not ct_path.exists():
        print(f"  CT not found: {ct_path}")
        return

    if not pet_reg_path.exists():
        print(f"  PET registered not found")
        print("  Skipping CT/PET visualization (CT only patient)")
        return

    print("Loading images...")
    ct_img = nib.load(ct_path)
    pet_img = nib.load(pet_reg_path)

    ct_data = ct_img.get_fdata()
    pet_data = pet_img.get_fdata()

    print(f"  CT shape: {ct_data.shape}")
    print(f"  PET shape: {pet_data.shape}")

    # 全肺葉のマスクを読み込み
    lobe_data = {}
    combined_mask = np.zeros_like(ct_data)
    for i, (lobe_name, (label, color)) in enumerate(LUNG_LOBES.items(), 1):
        mask_path = seg_dir / f"{lobe_name}.nii.gz"
        if mask_path.exists():
            mask = nib.load(mask_path).get_fdata()
            lobe_data[lobe_name] = {'data': mask, 'label': label, 'color': color, 'id': i}
            combined_mask[mask > 0] = i
            print(f"  Loaded: {lobe_name}")

    if not lobe_data:
        print("  No lung lobe segmentations found")
        return

    # マスクが存在するスライス範囲
    seg_z = np.where(np.any(combined_mask > 0, axis=(0, 1)))[0]
    z_min, z_max = seg_z.min(), seg_z.max()

    # 3スライス選択（上部、中央、下部）
    slices = [
        z_min + int((z_max - z_min) * 0.25),
        z_min + int((z_max - z_min) * 0.50),
        z_min + int((z_max - z_min) * 0.75),
    ]
    print(f"Selected slices: {slices}")

    # 凡例要素
    legend_elements = [plt.Line2D([0], [0], color=info['color'], linewidth=4, label=info['label'])
                       for info in lobe_data.values()]

    # ========================================
    # Figure 1: アキシャル断面（3スライス）
    # ========================================
    print("Generating axial views...")

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for row, z_idx in enumerate(slices):
        ct_slice = ct_data[:, :, z_idx].T
        pet_slice = pet_data[:, :, z_idx].T

        # CT + Segmentation
        ax = axes[row, 0]
        ax.imshow(ct_slice, cmap='gray', origin='lower', vmin=-1000, vmax=400)
        for lobe_name, info in lobe_data.items():
            mask_slice = info['data'][:, :, z_idx].T
            if np.any(mask_slice > 0):
                ax.contour(mask_slice, levels=[0.5], colors=[info['color']], linewidths=2)
        if row == 0:
            ax.set_title('CT + Segmentation', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'Slice {z_idx}', fontsize=12)
        ax.axis('off')

        # PET + Segmentation
        ax = axes[row, 1]
        vmax = np.percentile(pet_slice[pet_slice > 0], 98) if np.any(pet_slice > 0) else 1
        ax.imshow(pet_slice, cmap='hot', origin='lower', vmin=0, vmax=vmax)
        for lobe_name, info in lobe_data.items():
            mask_slice = info['data'][:, :, z_idx].T
            if np.any(mask_slice > 0):
                ax.contour(mask_slice, levels=[0.5], colors=[info['color']], linewidths=2)
        if row == 0:
            ax.set_title('PET + Segmentation', fontsize=14, fontweight='bold')
        ax.axis('off')

        # Fusion (CT + PET)
        ax = axes[row, 2]
        ax.imshow(ct_slice, cmap='gray', origin='lower', vmin=-1000, vmax=400)
        ax.imshow(pet_slice, cmap='hot', origin='lower', vmin=0, vmax=vmax, alpha=0.5)
        for lobe_name, info in lobe_data.items():
            mask_slice = info['data'][:, :, z_idx].T
            if np.any(mask_slice > 0):
                ax.contour(mask_slice, levels=[0.5], colors=[info['color']], linewidths=2)
        if row == 0:
            ax.set_title('CT/PET Fusion + Segmentation', fontsize=14, fontweight='bold')
        ax.axis('off')

    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=11,
               bbox_to_anchor=(0.5, 0.02))

    plt.suptitle(f'{patient_id} - CT/PET Registration Quality Check\n(Lung Lobe Segmentation Overlay)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    output_path = OUTPUT_DIR / f'{patient_id}_ct_pet_registration_axial.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

    # ========================================
    # Figure 2: コロナル断面
    # ========================================
    print("Generating coronal view...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 8))

    y_mid = ct_data.shape[1] // 2

    ct_coronal = ct_data[:, y_mid, :].T
    pet_coronal = pet_data[:, y_mid, :].T

    # CT coronal
    ax = axes[0]
    ax.imshow(ct_coronal, cmap='gray', origin='lower', vmin=-1000, vmax=400, aspect='auto')
    for lobe_name, info in lobe_data.items():
        mask_coronal = info['data'][:, y_mid, :].T
        if np.any(mask_coronal > 0):
            ax.contour(mask_coronal, levels=[0.5], colors=[info['color']], linewidths=2)
    ax.set_title('CT Coronal + Segmentation', fontsize=14, fontweight='bold')
    ax.axis('off')

    # PET coronal
    ax = axes[1]
    vmax = np.percentile(pet_coronal[pet_coronal > 0], 98) if np.any(pet_coronal > 0) else 1
    ax.imshow(pet_coronal, cmap='hot', origin='lower', vmin=0, vmax=vmax, aspect='auto')
    for lobe_name, info in lobe_data.items():
        mask_coronal = info['data'][:, y_mid, :].T
        if np.any(mask_coronal > 0):
            ax.contour(mask_coronal, levels=[0.5], colors=[info['color']], linewidths=2)
    ax.set_title('PET Coronal + Segmentation', fontsize=14, fontweight='bold')
    ax.axis('off')

    # Fusion coronal
    ax = axes[2]
    ax.imshow(ct_coronal, cmap='gray', origin='lower', vmin=-1000, vmax=400, aspect='auto')
    ax.imshow(pet_coronal, cmap='hot', origin='lower', vmin=0, vmax=vmax, alpha=0.5, aspect='auto')
    for lobe_name, info in lobe_data.items():
        mask_coronal = info['data'][:, y_mid, :].T
        if np.any(mask_coronal > 0):
            ax.contour(mask_coronal, levels=[0.5], colors=[info['color']], linewidths=2)
    ax.set_title('CT/PET Fusion Coronal', fontsize=14, fontweight='bold')
    ax.axis('off')

    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=11,
               bbox_to_anchor=(0.5, 0.02))
    plt.suptitle(f'{patient_id} - Coronal View - Registration Check', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    output_path = OUTPUT_DIR / f'{patient_id}_ct_pet_registration_coronal.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

    # ========================================
    # Figure 3: 腫瘍部位（SUV最大値位置）の断面
    # ========================================
    print("Generating tumor location view (SUV max)...")

    # 各肺葉内でのSUV最大値を探す
    tumor_info = find_tumor_location(pet_data, lobe_data)

    if tumor_info:
        generate_tumor_view(patient_id, ct_data, pet_data, lobe_data, tumor_info, legend_elements)


def find_tumor_location(pet_data, lobe_data):
    """各肺葉内のSUV最大値位置を特定"""
    tumor_info = []

    for lobe_name, info in lobe_data.items():
        mask = info['data']
        # マスク内のPET値を取得
        masked_pet = pet_data * (mask > 0)

        if np.any(masked_pet > 0):
            max_val = np.max(masked_pet)
            max_pos = np.unravel_index(np.argmax(masked_pet), masked_pet.shape)
            mean_val = np.mean(pet_data[mask > 0])

            # 左右を判定
            side = 'left' if 'left' in lobe_name else 'right'

            tumor_info.append({
                'lobe': lobe_name,
                'label': info['label'],
                'color': info['color'],
                'max_suv': max_val,
                'mean_suv': mean_val,
                'position': max_pos,  # (x, y, z)
                'side': side,
            })

    # SUV最大値でソート（降順）
    tumor_info.sort(key=lambda x: x['max_suv'], reverse=True)
    return tumor_info


def calculate_contralateral_lung_stats(pet_data, lobe_data, tumor_side):
    """腫瘍がない側（対側肺）の統計を計算"""
    # 対側を決定
    contralateral_side = 'left' if tumor_side == 'right' else 'right'

    # 対側肺のマスクを統合
    combined_mask = np.zeros_like(pet_data, dtype=bool)
    included_lobes = []

    for lobe_name, info in lobe_data.items():
        if contralateral_side in lobe_name:
            mask = info['data'] > 0
            combined_mask = combined_mask | mask
            included_lobes.append(info['label'])

    if not np.any(combined_mask):
        return None

    # 対側肺内のPET統計
    pet_in_mask = pet_data[combined_mask]

    stats = {
        'side': contralateral_side,
        'side_label': 'Left Lung' if contralateral_side == 'left' else 'Right Lung',
        'lobes': included_lobes,
        'mask': combined_mask,
        'voxel_count': np.sum(combined_mask),
        'suv_max': np.max(pet_in_mask),
        'suv_mean': np.mean(pet_in_mask),
        'suv_std': np.std(pet_in_mask),
        'suv_median': np.median(pet_in_mask),
        'suv_percentile_95': np.percentile(pet_in_mask, 95),
    }

    return stats


def generate_tumor_view(patient_id, ct_data, pet_data, lobe_data, tumor_info, legend_elements):
    """腫瘍部位（SUV最大値位置）の詳細画像を生成"""

    if not tumor_info:
        print("  No tumor location found")
        return

    # 最もSUVが高い部位
    primary = tumor_info[0]
    x, y, z = primary['position']

    print(f"  Primary lesion: {primary['label']}")
    print(f"    SUV max: {primary['max_suv']:.2f}")
    print(f"    Position: x={x}, y={y}, z={z}")

    # 3方向の断面を作成
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # SUV表示の上限
    pet_vmax = min(primary['max_suv'] * 1.2, np.percentile(pet_data[pet_data > 0], 99.5))

    # ========== 上段: CT + PET Fusion ==========

    # アキシャル（横断面）- 腫瘍位置
    ax = axes[0, 0]
    ct_slice = ct_data[:, :, z].T
    pet_slice = pet_data[:, :, z].T
    ax.imshow(ct_slice, cmap='gray', origin='lower', vmin=-1000, vmax=400)
    ax.imshow(pet_slice, cmap='hot', origin='lower', vmin=0, vmax=pet_vmax, alpha=0.6)
    # セグメンテーション輪郭
    for lobe_name, info in lobe_data.items():
        mask_slice = info['data'][:, :, z].T
        if np.any(mask_slice > 0):
            ax.contour(mask_slice, levels=[0.5], colors=[info['color']], linewidths=2)
    # 腫瘍位置にクロスヘア
    ax.axhline(y=x, color='lime', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=y, color='lime', linestyle='--', linewidth=1, alpha=0.7)
    ax.scatter([y], [x], c='lime', s=100, marker='+', linewidths=3)
    ax.set_title(f'Axial (z={z})', fontsize=14, fontweight='bold')
    ax.axis('off')

    # コロナル（冠状断面）- 腫瘍位置
    ax = axes[0, 1]
    ct_coronal = ct_data[:, y, :].T
    pet_coronal = pet_data[:, y, :].T
    ax.imshow(ct_coronal, cmap='gray', origin='lower', vmin=-1000, vmax=400, aspect='auto')
    ax.imshow(pet_coronal, cmap='hot', origin='lower', vmin=0, vmax=pet_vmax, alpha=0.6, aspect='auto')
    for lobe_name, info in lobe_data.items():
        mask_coronal = info['data'][:, y, :].T
        if np.any(mask_coronal > 0):
            ax.contour(mask_coronal, levels=[0.5], colors=[info['color']], linewidths=2)
    ax.axhline(y=z, color='lime', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=x, color='lime', linestyle='--', linewidth=1, alpha=0.7)
    ax.scatter([x], [z], c='lime', s=100, marker='+', linewidths=3)
    ax.set_title(f'Coronal (y={y})', fontsize=14, fontweight='bold')
    ax.axis('off')

    # サジタル（矢状断面）- 腫瘍位置
    ax = axes[0, 2]
    ct_sagittal = ct_data[x, :, :].T
    pet_sagittal = pet_data[x, :, :].T
    ax.imshow(ct_sagittal, cmap='gray', origin='lower', vmin=-1000, vmax=400, aspect='auto')
    ax.imshow(pet_sagittal, cmap='hot', origin='lower', vmin=0, vmax=pet_vmax, alpha=0.6, aspect='auto')
    for lobe_name, info in lobe_data.items():
        mask_sagittal = info['data'][x, :, :].T
        if np.any(mask_sagittal > 0):
            ax.contour(mask_sagittal, levels=[0.5], colors=[info['color']], linewidths=2)
    ax.axhline(y=z, color='lime', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=y, color='lime', linestyle='--', linewidth=1, alpha=0.7)
    ax.scatter([y], [z], c='lime', s=100, marker='+', linewidths=3)
    ax.set_title(f'Sagittal (x={x})', fontsize=14, fontweight='bold')
    ax.axis('off')

    # ========== 下段: PETのみ（カラーバー付き） ==========

    # アキシャル PET
    ax = axes[1, 0]
    im = ax.imshow(pet_slice, cmap='hot', origin='lower', vmin=0, vmax=pet_vmax)
    for lobe_name, info in lobe_data.items():
        mask_slice = info['data'][:, :, z].T
        if np.any(mask_slice > 0):
            ax.contour(mask_slice, levels=[0.5], colors=[info['color']], linewidths=2)
    ax.scatter([y], [x], c='lime', s=100, marker='+', linewidths=3)
    ax.set_title('PET Axial', fontsize=14, fontweight='bold')
    ax.axis('off')

    # コロナル PET
    ax = axes[1, 1]
    ax.imshow(pet_coronal, cmap='hot', origin='lower', vmin=0, vmax=pet_vmax, aspect='auto')
    for lobe_name, info in lobe_data.items():
        mask_coronal = info['data'][:, y, :].T
        if np.any(mask_coronal > 0):
            ax.contour(mask_coronal, levels=[0.5], colors=[info['color']], linewidths=2)
    ax.scatter([x], [z], c='lime', s=100, marker='+', linewidths=3)
    ax.set_title('PET Coronal', fontsize=14, fontweight='bold')
    ax.axis('off')

    # サジタル PET
    ax = axes[1, 2]
    ax.imshow(pet_sagittal, cmap='hot', origin='lower', vmin=0, vmax=pet_vmax, aspect='auto')
    for lobe_name, info in lobe_data.items():
        mask_sagittal = info['data'][x, :, :].T
        if np.any(mask_sagittal > 0):
            ax.contour(mask_sagittal, levels=[0.5], colors=[info['color']], linewidths=2)
    ax.scatter([y], [z], c='lime', s=100, marker='+', linewidths=3)
    ax.set_title('PET Sagittal', fontsize=14, fontweight='bold')
    ax.axis('off')

    # カラーバー
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('SUV', fontsize=12)

    # 対側肺（腫瘍がない側）の統計を計算
    contralateral = calculate_contralateral_lung_stats(pet_data, lobe_data, primary['side'])

    # SUV情報テキスト（腫瘍側）
    tumor_side_label = 'Right Lung' if primary['side'] == 'right' else 'Left Lung'
    info_text = f"=== TUMOR SIDE ({tumor_side_label}) ===\n"
    info_text += f"Primary Lesion: {primary['label']}\n"
    info_text += f"SUV max: {primary['max_suv']:.2f}\n"
    info_text += f"SUV mean: {primary['mean_suv']:.2f}\n"
    info_text += f"Position: ({x}, {y}, {z})"

    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.9))

    # 対側肺（正常側）の情報
    if contralateral:
        contra_text = f"=== CONTRALATERAL ({contralateral['side_label']}) ===\n"
        contra_text += f"Lobes: {', '.join(contralateral['lobes'])}\n"
        contra_text += f"SUV max:  {contralateral['suv_max']:.2f}\n"
        contra_text += f"SUV mean: {contralateral['suv_mean']:.2f}\n"
        contra_text += f"SUV std:  {contralateral['suv_std']:.2f}\n"
        contra_text += f"SUV median: {contralateral['suv_median']:.2f}\n"
        contra_text += f"SUV 95%ile: {contralateral['suv_percentile_95']:.2f}"

        fig.text(0.25, 0.02, contra_text, fontsize=10, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.9))

        print(f"  Contralateral ({contralateral['side_label']}):")
        print(f"    SUV mean: {contralateral['suv_mean']:.2f}")
        print(f"    SUV max:  {contralateral['suv_max']:.2f}")

    # 全肺葉のSUV一覧
    lobe_text = "SUV by Lobe:\n"
    for t in tumor_info:
        marker = "*" if t['lobe'] == primary['lobe'] else " "
        lobe_text += f"{marker} {t['label']}: max={t['max_suv']:.2f}, mean={t['mean_suv']:.2f}\n"

    fig.text(0.02, 0.88, lobe_text, fontsize=9, family='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # 凡例
    fig.legend(handles=legend_elements, loc='upper right', ncol=1, fontsize=9,
               bbox_to_anchor=(0.90, 0.88))

    contra_label = contralateral["side_label"] if contralateral else "N/A"
    plt.suptitle(f'{patient_id} - Tumor Location (SUV Maximum)\nTumor: {primary["label"]} | Contralateral: {contra_label}',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.12, 0.90, 0.94])

    output_path = OUTPUT_DIR / f'{patient_id}_tumor_location.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("CT/PET Registration Visualization (複数症例対応)")
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
