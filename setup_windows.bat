@echo off
REM ========================================
REM Lung PET Project - Windows Setup Script
REM KCRC Workstation
REM ========================================

echo ========================================
echo Lung PET/CT Radiomics Pipeline Setup
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Conda not found. Please install Anaconda or Miniconda first.
    echo Download: https://docs.anaconda.com/miniconda/
    pause
    exit /b 1
)

REM Check if environment already exists
conda env list | findstr "radiomics" >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Environment 'radiomics' already exists.
    set /p RECREATE="Do you want to recreate it? (y/N): "
    if /i "%RECREATE%"=="y" (
        echo Removing existing environment...
        conda env remove -n radiomics -y
    ) else (
        echo Skipping environment creation.
        goto :install_packages
    )
)

REM Create conda environment
echo.
echo [1/4] Creating conda environment 'radiomics'...
conda create -n radiomics python=3.10 -y
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create conda environment.
    pause
    exit /b 1
)

:install_packages
REM Activate environment
echo.
echo [2/4] Activating environment...
call conda activate radiomics

REM Install PyTorch with CUDA
echo.
echo [3/4] Installing PyTorch with CUDA 12.1...
echo (This may take a few minutes...)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo [WARNING] Failed to install CUDA PyTorch. Trying CPU version...
    pip install torch torchvision torchaudio
)

REM Install other dependencies
echo.
echo [4/4] Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

REM Create config.yaml if not exists
if not exist config.yaml (
    echo.
    echo Creating config.yaml from template...
    copy config.yaml.example config.yaml
)

REM Create data directories if not exist
if not exist raw_download mkdir raw_download
if not exist nifti_images mkdir nifti_images
if not exist segmentations mkdir segmentations
if not exist visualizations mkdir visualizations
if not exist analysis_results mkdir analysis_results

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Place DICOM data in 'raw_download' folder
echo   2. Edit config.yaml if needed
echo   3. Run: run_analysis.bat
echo.
echo To activate the environment manually:
echo   conda activate radiomics
echo.
pause
