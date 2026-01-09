"""
Mask Verification Visualization
==============================

縦隔条件のPET-CT融合画像にマスクをオーバーレイして表示
Axial + Coronal の2面で表示

臓器:
- 肝臓 (liver)
- 両側腎 (kidney_left, kidney_right)
- 両側副腎 (adrenal_gland_left, adrenal_gland_right)
- L1椎体 (vertebrae_L1)
- 大動脈 (aorta)
- 左肺 (lung_upper_lobe_left, lung_lower_lobe_left)
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import yaml


def load_config(config_path="config.yaml"):
    """設定ファイルの読み込み"""
    default_config = {
        'visualization': {
            'ct_window': {'level': 40, 'width': 400},  # 縦隔条件
            'pet_colormap': 'hot',
            'pet_suv_range': {'min': 0, 'max': 10},
            'mask_alpha': 0.3
        }
    }
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            if user_config and 'visualization' in user_config:
                default_config['visualization'].update(user_config['visualization'])
    return default_config


def apply_ct_window(ct_data, level=40, width=400):
    """
    CTデータにウィンドウ設定を適用

    デフォルト: 縦隔条件 (WL=40, WW=400)
    """
    min_val = level - width / 2
    max_val = level + width / 2
    windowed = np.clip(ct_data, min_val, max_val)
    windowed = (windowed - min_val) / (max_val - min_val)
    return windowed


def create_organ_colormap():
    """臓器ごとの色を定義"""
    # 臓器名と色の対応
    organ_colors = {
        'liver': [0.8, 0.4, 0.0, 0.6],          # オレンジ
        'kidney_left': [0.0, 0.6, 0.8, 0.6],     # シアン
        'kidney_right': [0.0, 0.8, 0.6, 0.6],    # ティール
        'adrenal_gland_left': [0.8, 0.8, 0.0, 0.6],   # 黄
        'adrenal_gland_right': [1.0, 0.6, 0.0, 0.6],  # オレンジ黄
        'vertebrae_L1': [0.6, 0.0, 0.8, 0.6],    # 紫
        'aorta': [0.8, 0.0, 0.0, 0.6],           # 赤
        'lung_upper_lobe_left': [0.0, 0.8, 0.0, 0.4],  # 緑（薄め）
        'lung_lower_lobe_left': [0.0, 0.5, 0.0, 0.4],  # 濃い緑（薄め）
    }
    return organ_colors


def load_masks(seg_dir, organs):
    """マスクファイルを読み込み"""
    masks = {}
    for organ in organs:
        mask_path = seg_dir / f"{organ}.nii.gz"
        if mask_path.exists():
            masks[organ] = nib.load(str(mask_path)).get_fdata()
    return masks


def create_mask_verification(case_id, nifti_dir=None, seg_dir=None, output_dir=None):
    """
    マスク検証画像を作成

    Parameters:
    -----------
    case_id : str
        患者ID (例: "ILD_001")
    nifti_dir : Path, optional
        NIfTIファイルのディレクトリ
    seg_dir : Path, optional
        セグメンテーションファイルのディレクトリ
    output_dir : Path, optional
        出力ディレクトリ
    """
    # デフォルトパス設定
    root_dir = Path(__file__).parent.resolve()
    if nifti_dir is None:
        nifti_dir = root_dir / "nifti_images"
    if seg_dir is None:
        seg_dir = root_dir / "segmentations" / f"{case_id}_CT"
    if output_dir is None:
        output_dir = root_dir / "visualizations" / case_id

    output_dir.mkdir(parents=True, exist_ok=True)

    # 設定読み込み
    config = load_config()
    viz_config = config['visualization']

    # ファイルパス
    ct_path = nifti_dir / f"{case_id}_CT.nii.gz"
    pet_path = nifti_dir / f"{case_id}_PET_registered.nii.gz"
    if not pet_path.exists():
        pet_path = nifti_dir / f"{case_id}_PET.nii.gz"

    # データ読み込み
    print(f"Loading CT: {ct_path}")
    ct_img = nib.load(str(ct_path))
    ct_data = ct_img.get_fdata()

    pet_data = None
    if pet_path.exists():
        print(f"Loading PET: {pet_path}")
        pet_img = nib.load(str(pet_path))
        pet_data = pet_img.get_fdata()
        # SUV正規化
        suv_min = viz_config['pet_suv_range']['min']
        suv_max = viz_config['pet_suv_range']['max']
        pet_normalized = np.clip(pet_data, suv_min, suv_max)
        pet_normalized = (pet_normalized - suv_min) / (suv_max - suv_min)

    # マスク読み込み
    organs = [
        'liver',
        'kidney_left', 'kidney_right',
        'adrenal_gland_left', 'adrenal_gland_right',
        'vertebrae_L1',
        'aorta',
        'lung_upper_lobe_left', 'lung_lower_lobe_left'
    ]
    masks = load_masks(seg_dir, organs)
    organ_colors = create_organ_colormap()

    print(f"Loaded masks: {list(masks.keys())}")

    # CT画像に縦隔条件を適用
    ct_level = viz_config['ct_window']['level']
    ct_width = viz_config['ct_window']['width']
    ct_windowed = apply_ct_window(ct_data, level=ct_level, width=ct_width)

    # 画像の形状
    shape = ct_data.shape
    print(f"Image shape: {shape}")

    # スライス位置の決定（各臓器の中心を含むスライス）
    slice_positions = find_representative_slices(masks, shape)

    # Axial + Coronal 画像を作成
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'PET-CT Mask Verification - {case_id}\n'
                 f'CT Window: Mediastinal (WL={ct_level}, WW={ct_width})',
                 fontsize=14, fontweight='bold')

    # Axial スライス（3枚）
    axial_slices = slice_positions.get('axial', [shape[2]//4, shape[2]//2, shape[2]*3//4])
    for i, z in enumerate(axial_slices[:3]):
        ax = axes[0, i]
        z = min(z, shape[2]-1)

        # PET-CT融合画像
        ct_slice = ct_windowed[:, :, z]
        if pet_data is not None:
            pet_slice = pet_normalized[:, :, z]
            fused = create_fused_image(ct_slice, pet_slice)
        else:
            fused = np.stack([ct_slice]*3, axis=-1)

        # 画像表示（回転して正しい向きに）
        fused_rotated = np.rot90(fused)
        ax.imshow(fused_rotated, origin='lower')

        # マスクオーバーレイ
        for organ, mask_data in masks.items():
            if organ in organ_colors:
                mask_slice = np.rot90(mask_data[:, :, z])
                color = organ_colors[organ]
                overlay_mask(ax, mask_slice, color)

        ax.set_title(f'Axial (z={z})', fontsize=11)
        ax.axis('off')

    # Coronal スライス（3枚）
    coronal_slices = slice_positions.get('coronal', [shape[1]//4, shape[1]//2, shape[1]*3//4])
    for i, y in enumerate(coronal_slices[:3]):
        ax = axes[1, i]
        y = min(y, shape[1]-1)

        # PET-CT融合画像
        ct_slice = ct_windowed[:, y, :]
        if pet_data is not None:
            pet_slice = pet_normalized[:, y, :]
            fused = create_fused_image(ct_slice, pet_slice)
        else:
            fused = np.stack([ct_slice]*3, axis=-1)

        # 画像表示
        fused_rotated = np.rot90(fused)
        ax.imshow(fused_rotated, origin='lower')

        # マスクオーバーレイ
        for organ, mask_data in masks.items():
            if organ in organ_colors:
                mask_slice = np.rot90(mask_data[:, y, :])
                color = organ_colors[organ]
                overlay_mask(ax, mask_slice, color)

        ax.set_title(f'Coronal (y={y})', fontsize=11)
        ax.axis('off')

    # 凡例追加
    add_legend(fig, organ_colors, masks.keys())

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    # 保存
    output_path = output_dir / "mask_verification.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def find_representative_slices(masks, shape):
    """臓器の位置からスライス位置を決定"""
    positions = {'axial': [], 'coronal': []}

    # 肝臓の中心
    if 'liver' in masks:
        z_indices = np.where(masks['liver'].sum(axis=(0, 1)) > 0)[0]
        if len(z_indices) > 0:
            positions['axial'].append(int(np.median(z_indices)))

    # 腎臓の中心
    for kidney in ['kidney_left', 'kidney_right']:
        if kidney in masks:
            z_indices = np.where(masks[kidney].sum(axis=(0, 1)) > 0)[0]
            if len(z_indices) > 0:
                positions['axial'].append(int(np.median(z_indices)))
                break

    # L1椎体の中心
    if 'vertebrae_L1' in masks:
        z_indices = np.where(masks['vertebrae_L1'].sum(axis=(0, 1)) > 0)[0]
        if len(z_indices) > 0:
            positions['axial'].append(int(np.median(z_indices)))

    # Coronalスライス: 大動脈、腎臓、左肺を含む位置
    if 'aorta' in masks:
        y_indices = np.where(masks['aorta'].sum(axis=(0, 2)) > 0)[0]
        if len(y_indices) > 0:
            positions['coronal'].append(int(np.median(y_indices)))

    for kidney in ['kidney_left', 'kidney_right']:
        if kidney in masks:
            y_indices = np.where(masks[kidney].sum(axis=(0, 2)) > 0)[0]
            if len(y_indices) > 0:
                positions['coronal'].append(int(np.median(y_indices)))

    if 'lung_upper_lobe_left' in masks:
        y_indices = np.where(masks['lung_upper_lobe_left'].sum(axis=(0, 2)) > 0)[0]
        if len(y_indices) > 0:
            positions['coronal'].append(int(np.median(y_indices)))

    # デフォルト値で補完
    if len(positions['axial']) < 3:
        positions['axial'].extend([shape[2]//4, shape[2]//2, shape[2]*3//4])
    if len(positions['coronal']) < 3:
        positions['coronal'].extend([shape[1]//4, shape[1]//2, shape[1]*3//4])

    # 重複除去と並び替え
    positions['axial'] = sorted(list(set(positions['axial'])))[:3]
    positions['coronal'] = sorted(list(set(positions['coronal'])))[:3]

    return positions


def create_fused_image(ct_slice, pet_slice, pet_alpha=0.5):
    """CT画像とPET画像を融合"""
    # CTをグレースケールのRGB
    ct_rgb = np.stack([ct_slice] * 3, axis=-1)

    # PETをホットカラーマップで変換
    cmap = plt.cm.hot
    pet_colored = cmap(pet_slice)[:, :, :3]

    # 融合（PETを重ねる）
    mask = pet_slice > 0.1  # 低値をマスク
    fused = ct_rgb.copy()
    fused[mask] = (1 - pet_alpha) * ct_rgb[mask] + pet_alpha * pet_colored[mask]

    return np.clip(fused, 0, 1)


def overlay_mask(ax, mask_slice, color):
    """マスクをオーバーレイ"""
    if mask_slice.sum() == 0:
        return

    # マスクの輪郭を描画
    from matplotlib.colors import to_rgba
    rgba = to_rgba(color[:3], alpha=color[3])

    mask_binary = mask_slice > 0
    overlay = np.zeros((*mask_slice.shape, 4))
    overlay[mask_binary] = rgba

    ax.imshow(overlay, origin='lower')


def add_legend(fig, organ_colors, available_organs):
    """凡例を追加"""
    # 日本語臓器名マッピング
    organ_names_jp = {
        'liver': 'Liver (肝)',
        'kidney_left': 'L Kidney (左腎)',
        'kidney_right': 'R Kidney (右腎)',
        'adrenal_gland_left': 'L Adrenal (左副腎)',
        'adrenal_gland_right': 'R Adrenal (右副腎)',
        'vertebrae_L1': 'L1 Vertebra (L1椎体)',
        'aorta': 'Aorta (大動脈)',
        'lung_upper_lobe_left': 'L Upper Lung (左肺上葉)',
        'lung_lower_lobe_left': 'L Lower Lung (左肺下葉)',
    }

    legend_elements = []
    for organ in organ_colors:
        if organ in available_organs:
            color = organ_colors[organ][:3]
            name = organ_names_jp.get(organ, organ)
            from matplotlib.patches import Patch
            legend_elements.append(Patch(facecolor=color, edgecolor='white',
                                        label=name, alpha=0.7))

    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
              fontsize=9, framealpha=0.8, facecolor='gray',
              edgecolor='white', labelcolor='white')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create mask verification visualization")
    parser.add_argument("case_id", type=str, help="Patient ID (e.g., ILD_001)")
    parser.add_argument("--nifti-dir", type=str, default=None, help="NIfTI directory")
    parser.add_argument("--seg-dir", type=str, default=None, help="Segmentation directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    create_mask_verification(
        case_id=args.case_id,
        nifti_dir=Path(args.nifti_dir) if args.nifti_dir else None,
        seg_dir=Path(args.seg_dir) if args.seg_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )
