#!/bin/bash
# ============================================================
# PET/CT Radiomics GUI ランチャー
# ダブルクリックで実行できます
# ============================================================

cd "$(dirname "$0")"

# Python環境でGUIを起動
/Users/akira/miniforge3/envs/med_ai/bin/python gui_launcher.py
