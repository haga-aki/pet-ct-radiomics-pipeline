@echo off
REM ============================================
REM PET/CT Radiomics Pipeline - Windows Launcher
REM ============================================
REM
REM Usage:
REM   run_analysis.bat              - Full analysis
REM   run_analysis.bat --force      - Force reprocess all
REM   run_analysis.bat --visualize-only  - Visualization only
REM
REM Environment: radiomics (Conda)
REM ============================================

chcp 65001 > nul
setlocal enabledelayedexpansion

REM Conda path (adjust if different)
set CONDA_PATH=D:\Haga\Apps\miniconda3
set ENV_NAME=radiomics

echo.
echo ============================================
echo  PET/CT Radiomics Pipeline
echo ============================================
echo.

REM Activate conda environment
call "%CONDA_PATH%\Scripts\activate.bat" %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment: %ENV_NAME%
    echo Please run: conda env create -f environment_windows.yml
    pause
    exit /b 1
)

echo [OK] Conda environment: %ENV_NAME%
echo.

REM Run the pipeline
python run_full_analysis.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline failed with error code %errorlevel%
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Complete!
echo ============================================
pause
