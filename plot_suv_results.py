#!/Users/akira/miniforge3/envs/med_ai/bin/python
"""
補正後のPET SUV Radiomics結果を可視化
複数症例対応版: 各患者ごとにグラフを生成
"""
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ディレクトリ設定
BASE_DIR = Path("/Users/akira/Local/medical_ai/ycu_project")
OUTPUT_DIR = BASE_DIR / "analysis_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# 肺葉の順序と表示名
ORGAN_ORDER = [
    'lung_upper_lobe_left',
    'lung_lower_lobe_left',
    'lung_upper_lobe_right',
    'lung_middle_lobe_right',
    'lung_lower_lobe_right'
]
ORGAN_LABELS = ['Left Upper', 'Left Lower', 'Right Upper', 'Right Middle', 'Right Lower']
COLORS = ['#3498db', '#2980b9', '#e74c3c', '#c0392b', '#d35400']


def plot_patient_suv(df_pet, patient_id, output_dir):
    """1患者のSUVグラフを生成"""

    # データを順序通りに並べ替え
    df_pet = df_pet.copy()
    df_pet['order'] = df_pet['Organ'].map({o: i for i, o in enumerate(ORGAN_ORDER)})
    df_pet = df_pet.sort_values('order')

    # 統計量を抽出
    means = df_pet['original_firstorder_Mean'].values
    maxs = df_pet['original_firstorder_Maximum'].values
    mins = df_pet['original_firstorder_Minimum'].values
    stds = np.sqrt(df_pet['original_firstorder_Variance'].values)
    medians = df_pet['original_firstorder_Median'].values
    p10 = df_pet['original_firstorder_10Percentile'].values
    p90 = df_pet['original_firstorder_90Percentile'].values
    skewness = df_pet['original_firstorder_Skewness'].values
    kurtosis = df_pet['original_firstorder_Kurtosis'].values

    # Figure 1: 主要なSUV統計量
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'PET SUV Analysis - {patient_id}', fontsize=16, fontweight='bold')

    # Mean SUV
    ax = axes[0, 0]
    bars = ax.bar(ORGAN_LABELS, means, color=COLORS, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('SUVbw', fontsize=12)
    ax.set_title('Mean SUV by Lung Lobe', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(means) * 1.3 if max(means) > 0 else 1)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Normal lung range')
    ax.tick_params(axis='x', rotation=30)

    # Max SUV
    ax = axes[0, 1]
    bars = ax.bar(ORGAN_LABELS, maxs, color=COLORS, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('SUVbw', fontsize=12)
    ax.set_title('Maximum SUV by Lung Lobe', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(maxs) * 1.2 if max(maxs) > 0 else 1)
    for bar, val in zip(bars, maxs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.axhline(y=2.5, color='orange', linestyle='--', alpha=0.7, label='SUV 2.5 threshold')
    ax.legend(loc='upper left')
    ax.tick_params(axis='x', rotation=30)

    # SUV Distribution (Box-like visualization)
    ax = axes[1, 0]
    x = np.arange(len(ORGAN_LABELS))
    width = 0.6

    for i, (label, color) in enumerate(zip(ORGAN_LABELS, COLORS)):
        # Box (10th to 90th percentile)
        ax.bar(i, p90[i] - p10[i], bottom=p10[i], width=width, color=color, alpha=0.7, edgecolor='black')
        # Median line
        ax.hlines(medians[i], i - width/2, i + width/2, colors='white', linewidth=2)
        # Min-Max whiskers
        ax.vlines(i, mins[i], p10[i], colors='black', linewidth=1)
        ax.vlines(i, p90[i], maxs[i], colors='black', linewidth=1)
        ax.hlines(mins[i], i - 0.1, i + 0.1, colors='black', linewidth=1)
        ax.hlines(maxs[i], i - 0.1, i + 0.1, colors='black', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(ORGAN_LABELS, rotation=30)
    ax.set_ylabel('SUVbw', fontsize=12)
    ax.set_title('SUV Distribution\n(Box: 10th-90th %ile, Line: Median, Whiskers: Min-Max)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(maxs) * 1.1 if max(maxs) > 0 else 1)

    # Standard Deviation
    ax = axes[1, 1]
    bars = ax.bar(ORGAN_LABELS, stds, color=COLORS, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Standard Deviation', fontsize=12)
    ax.set_title('SUV Standard Deviation by Lung Lobe', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    output_path = output_dir / f'{patient_id}_pet_suv_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

    # Figure 2: 詳細統計量
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'PET SUV Detailed Statistics - {patient_id}', fontsize=14, fontweight='bold')

    # Skewness
    ax = axes[0]
    bars = ax.bar(ORGAN_LABELS, skewness, color=COLORS, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Skewness', fontsize=12)
    ax.set_title('SUV Skewness\n(Distribution Asymmetry)', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    for bar, val in zip(bars, skewness):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)

    # Kurtosis
    ax = axes[1]
    bars = ax.bar(ORGAN_LABELS, kurtosis, color=COLORS, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Kurtosis', fontsize=12)
    ax.set_title('SUV Kurtosis\n(Distribution Peakedness)', fontsize=12, fontweight='bold')
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Normal distribution')
    ax.legend()
    for bar, val in zip(bars, kurtosis):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)

    # Coefficient of Variation (CV)
    cv = stds / means * 100
    ax = axes[2]
    bars = ax.bar(ORGAN_LABELS, cv, color=COLORS, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('CV (%)', fontsize=12)
    ax.set_title('Coefficient of Variation\n(Heterogeneity Index)', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, cv):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    output_path = output_dir / f'{patient_id}_pet_suv_details.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

    # Figure 3: サマリーテーブル画像
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    table_data = []
    for i, organ in enumerate(ORGAN_LABELS):
        table_data.append([
            organ,
            f'{means[i]:.3f}',
            f'{medians[i]:.3f}',
            f'{maxs[i]:.2f}',
            f'{mins[i]:.3f}',
            f'{stds[i]:.3f}',
            f'{skewness[i]:.2f}',
            f'{kurtosis[i]:.1f}'
        ])

    columns = ['Lung Lobe', 'Mean', 'Median', 'Max', 'Min', 'Std', 'Skewness', 'Kurtosis']

    table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # ヘッダーの色
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Max SUVが最大の行をハイライト
    max_idx = np.argmax(maxs) + 1  # +1 for header row
    for i in range(len(columns)):
        table[(max_idx, i)].set_facecolor('#FFE6E6')

    ax.set_title(f'PET SUVbw Radiomics Summary ({patient_id})\n(Highest Max SUV lobe highlighted)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = output_dir / f'{patient_id}_pet_suv_table.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_comparison(df_pet, output_dir):
    """複数患者の比較グラフを生成"""
    if df_pet['PatientID'].nunique() < 2:
        return

    patient_ids = sorted(df_pet['PatientID'].unique())
    n_patients = len(patient_ids)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Patient Comparison - PET SUV', fontsize=16, fontweight='bold')

    x = np.arange(len(ORGAN_LABELS))
    width = 0.8 / n_patients

    # Mean SUV comparison
    ax = axes[0]
    for i, pid in enumerate(patient_ids):
        df_p = df_pet[df_pet['PatientID'] == pid].copy()
        df_p['order'] = df_p['Organ'].map({o: j for j, o in enumerate(ORGAN_ORDER)})
        df_p = df_p.sort_values('order')
        means = df_p['original_firstorder_Mean'].values
        ax.bar(x + i * width - width * (n_patients-1) / 2, means, width, label=pid)

    ax.set_xlabel('Lung Lobe')
    ax.set_ylabel('Mean SUVbw')
    ax.set_title('Mean SUV Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(ORGAN_LABELS, rotation=30)
    ax.legend()

    # Max SUV comparison
    ax = axes[1]
    for i, pid in enumerate(patient_ids):
        df_p = df_pet[df_pet['PatientID'] == pid].copy()
        df_p['order'] = df_p['Organ'].map({o: j for j, o in enumerate(ORGAN_ORDER)})
        df_p = df_p.sort_values('order')
        maxs = df_p['original_firstorder_Maximum'].values
        ax.bar(x + i * width - width * (n_patients-1) / 2, maxs, width, label=pid)

    ax.set_xlabel('Lung Lobe')
    ax.set_ylabel('Max SUVbw')
    ax.set_title('Max SUV Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(ORGAN_LABELS, rotation=30)
    ax.axhline(y=2.5, color='orange', linestyle='--', alpha=0.7, label='SUV 2.5')
    ax.legend()

    plt.tight_layout()
    output_path = output_dir / 'all_patients_suv_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("PET SUV Results Visualization (複数症例対応)")
    print("=" * 70)

    # データ読み込み
    csv_path = BASE_DIR / "pet_ct_radiomics_results.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    df_pet = df[df['Modality'] == 'PET'].copy()

    if df_pet.empty:
        print("No PET data found in CSV")
        return

    patient_ids = df_pet['PatientID'].unique()
    print(f"\n対象患者数: {len(patient_ids)}")

    # 各患者のグラフを生成
    for patient_id in sorted(patient_ids):
        print(f"\n--- {patient_id} ---")
        df_patient = df_pet[df_pet['PatientID'] == patient_id]
        plot_patient_suv(df_patient, patient_id, OUTPUT_DIR)

    # 複数患者の比較グラフ
    if len(patient_ids) > 1:
        print("\n--- 患者比較グラフ ---")
        plot_comparison(df_pet, OUTPUT_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
