@echo off

set paths=%ProgramData%\miniconda3\Scripts
set paths=%paths%;%USERPROFILE%\miniconda3\Scripts
set paths=%paths%;%ProgramData%\anaconda3\Scripts
set paths=%paths%;%USERPROFILE%\anaconda3\Scripts

for %%a in (%paths%) do ( 
 if EXIST "%%a\activate.bat" (
    SET CONDA_PATH=%%a
 )
)

IF "%CONDA_PATH%"=="" (
  echo anaconda3/miniconda3 not found. Install from here https://docs.conda.io/en/latest/miniconda.html
  exit /b 1 
) else (
  echo anaconda3/miniconda3 detected in %CONDA_PATH%
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

