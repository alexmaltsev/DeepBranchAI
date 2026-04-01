@echo off
echo === DeepBranchAI Installation ===
echo.

REM Locate conda installation
set "CONDA_BASE="
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
    set "CONDA_BASE=%USERPROFILE%\miniconda3"
) else if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
    set "CONDA_BASE=%USERPROFILE%\anaconda3"
) else if exist "%PROGRAMDATA%\miniconda3\condabin\conda.bat" (
    set "CONDA_BASE=%PROGRAMDATA%\miniconda3"
) else (
    echo ERROR: conda not found. Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)
echo Found conda at: %CONDA_BASE%

REM Initialize conda for this shell session
call "%CONDA_BASE%\condabin\conda.bat" activate base
if errorlevel 1 (
    echo ERROR: Failed to initialize conda.
    pause
    exit /b 1
)

REM Create conda environment (skip if it already exists)
echo.
echo Creating conda environment 'deepbranchai'...
call conda create -n deepbranchai python=3.12 -y
if errorlevel 1 (
    echo ERROR: Failed to create conda environment.
    pause
    exit /b 1
)

REM Activate the environment
call conda activate deepbranchai
if errorlevel 1 (
    echo ERROR: Failed to activate deepbranchai environment.
    pause
    exit /b 1
)

REM Verify we are in the right environment
echo.
echo Active Python:
python -c "import sys; print(sys.executable)"

REM Install PyTorch with CUDA
echo.
echo Installing PyTorch with CUDA...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo ERROR: PyTorch installation failed.
    pause
    exit /b 1
)

REM Install remaining dependencies
echo.
echo Installing dependencies...
python -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
)

REM Verify key imports
echo.
echo Verifying installation...
python -c "import torch; print('PyTorch', torch.__version__, '- CUDA:', torch.cuda.is_available())"
python -c "import nnunetv2; print('nnU-Net v2 OK')"
python -c "import nibabel; print('nibabel OK')"
python -c "import jupyter; print('jupyter OK')"

echo.
echo === Installation complete ===
echo.
echo To use DeepBranchAI, open Anaconda Prompt and run:
echo.
echo     conda activate deepbranchai
echo     jupyter notebook
echo.
pause
