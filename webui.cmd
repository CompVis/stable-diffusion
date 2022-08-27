@echo off
set CONDA_ALL=%ProgramData%\miniconda3\Scripts
set CONDA_USER=%USERPROFILE%\miniconda3\Scripts

IF EXIST %CONDA_ALL% (
  SET CONDA_PATH=%CONDA_ALL%
) else IF EXIST %CONDA_USER% (
   SET CONDA_PATH=%CONDA_USER% 
) else (
  echo "miniconda3 not found. Install from here https://docs.conda.io/en/latest/miniconda.html"
  exit /b 1 
)

call "%CONDA_PATH%\activate.bat"
call conda env create -f environment.yaml
call conda env update --file environment.yaml --prune
call "%CONDA_PATH%\activate.bat" ldo
python "%CD%"\scripts\relauncher.py

:PROMPT
set SETUPTOOLS_USE_DISTUTILS=stdlib
IF EXIST "models\ldm\stable-diffusion-v1\model.ckpt" (
  python scripts/relauncher.py
) ELSE (
  ECHO Your model file does not exist! Place it in 'models\ldm\stable-diffusion-v1' with the name 'model.ckpt'.
)
