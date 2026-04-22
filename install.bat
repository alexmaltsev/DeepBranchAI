@echo off
setlocal

set "ENV_NAME=deepbranchai"
set "SCRIPT_DIR=%~dp0"
set "ERROR_MSG="

pushd "%SCRIPT_DIR%" >nul

echo === DeepBranchAI Installation ===
echo.

call :find_conda
if errorlevel 1 goto :fail
echo Found conda at: %CONDA_BAT%

call "%CONDA_BAT%" activate base
if errorlevel 1 (
    set "ERROR_MSG=Failed to initialize conda."
    goto :fail
)

echo.
echo Checking conda environment '%ENV_NAME%'...
call "%CONDA_BAT%" run -n "%ENV_NAME%" python -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo Creating conda environment '%ENV_NAME%'...
    call "%CONDA_BAT%" create -n "%ENV_NAME%" python=3.12 -y
    if errorlevel 1 (
        set "ERROR_MSG=Failed to create conda environment."
        goto :fail
    )
) else (
    echo Reusing existing conda environment '%ENV_NAME%'.
)

call "%CONDA_BAT%" activate "%ENV_NAME%"
if errorlevel 1 (
    set "ERROR_MSG=Failed to activate %ENV_NAME% environment."
    goto :fail
)

echo.
echo Active Python:
python -c "import sys; print(sys.executable)"
if errorlevel 1 (
    set "ERROR_MSG=Python is not available inside the activated environment."
    goto :fail
)

echo.
echo Upgrading pip tooling...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    set "ERROR_MSG=Failed to upgrade pip tooling."
    goto :fail
)

echo.
echo Installing PyTorch with CUDA 11.8...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    set "ERROR_MSG=PyTorch installation failed."
    goto :fail
)

echo.
echo Installing DeepBranchAI dependencies...
python -m pip install -r "%SCRIPT_DIR%requirements.txt"
if errorlevel 1 (
    set "ERROR_MSG=Dependency installation failed."
    goto :fail
)

echo.
echo Registering Jupyter kernel...
python -m ipykernel install --user --name "%ENV_NAME%" --display-name "%ENV_NAME%"
if errorlevel 1 (
    set "ERROR_MSG=Jupyter kernel registration failed."
    goto :fail
)

echo.
echo Verifying installation...
python -c "import torch, nnunetv2, nibabel, jupyter; print('PyTorch', torch.__version__, '- CUDA:', torch.cuda.is_available()); print('nnU-Net v2 OK'); print('nibabel OK'); print('jupyter OK')"
if errorlevel 1 (
    set "ERROR_MSG=Installation verification failed."
    goto :fail
)

echo.
echo === Installation complete ===
echo.
echo To use DeepBranchAI, open Anaconda Prompt and run:
echo.
echo     conda activate %ENV_NAME%
echo     jupyter notebook
echo.
goto :success

:find_conda
set "CONDA_BAT="

if defined CONDA_EXE (
    if exist "%CONDA_EXE%" (
        for %%I in ("%CONDA_EXE%") do (
            if /I "%%~xI"==".bat" set "CONDA_BAT=%%~fI"
            if /I "%%~nxI"=="conda.exe" if exist "%%~dpI..\condabin\conda.bat" set "CONDA_BAT=%%~dpI..\condabin\conda.bat"
        )
    )
)

if not defined CONDA_BAT if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
    set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
)
if not defined CONDA_BAT if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" (
    set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
)
if not defined CONDA_BAT if exist "%PROGRAMDATA%\miniconda3\condabin\conda.bat" (
    set "CONDA_BAT=%PROGRAMDATA%\miniconda3\condabin\conda.bat"
)
if not defined CONDA_BAT if exist "%PROGRAMDATA%\anaconda3\condabin\conda.bat" (
    set "CONDA_BAT=%PROGRAMDATA%\anaconda3\condabin\conda.bat"
)
if not defined CONDA_BAT (
    for /f "delims=" %%I in ('where conda.bat 2^>nul') do if not defined CONDA_BAT set "CONDA_BAT=%%I"
)

if not defined CONDA_BAT (
    set "ERROR_MSG=conda not found. Install Miniconda from https://docs.conda.io/en/latest/miniconda.html"
    exit /b 1
)

exit /b 0

:fail
echo.
echo ERROR: %ERROR_MSG%
popd >nul
pause
exit /b 1

:success
popd >nul
pause
exit /b 0
