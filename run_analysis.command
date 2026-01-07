#!/bin/bash
# ============================================================
# PET/CT Radiomics 解析ランチャー (macOS)
# ダブルクリックで実行できます
# ============================================================

cd "$(dirname "$0")"

echo "============================================================"
echo "  PET/CT Radiomics 統合解析パイプライン"
echo "============================================================"
echo ""
echo "作業ディレクトリ: $(pwd)"
echo ""

# Python環境
PYTHON="/Users/akira/miniforge3/envs/med_ai/bin/python"

# 環境確認
if [ ! -f "$PYTHON" ]; then
    echo "エラー: Python環境が見つかりません"
    echo "パス: $PYTHON"
    read -p "Press Enter to exit..."
    exit 1
fi

# メニュー表示
echo "実行オプションを選択してください:"
echo ""
echo "  1) フル解析（全自動）"
echo "  2) 可視化のみ"
echo "  3) 新規データのみ処理"
echo "  4) 強制再処理（全データ）"
echo "  5) 終了"
echo ""
read -p "選択 (1-5): " choice

case $choice in
    1)
        echo ""
        echo "フル解析を実行します..."
        $PYTHON run_full_analysis.py
        ;;
    2)
        echo ""
        echo "可視化のみ実行します..."
        $PYTHON run_full_analysis.py --visualize-only
        ;;
    3)
        echo ""
        echo "新規データのみ処理します..."
        $PYTHON run_full_analysis.py
        ;;
    4)
        echo ""
        echo "全データを強制再処理します..."
        $PYTHON run_full_analysis.py --force
        ;;
    5)
        echo "終了します"
        exit 0
        ;;
    *)
        echo "無効な選択です"
        ;;
esac

echo ""
echo "============================================================"
echo "  処理が完了しました"
echo "============================================================"
echo ""
echo "結果ファイル:"
echo "  - pet_ct_radiomics_results.csv"
echo "  - analysis_results/ (画像)"
echo ""
read -p "Press Enter to exit..."
