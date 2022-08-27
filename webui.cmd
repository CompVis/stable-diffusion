@echo off
call C:\ProgramData\miniconda3\Scripts\activate.bat
call conda env create -f environment.yaml
call conda env update --file environment.yaml --prune
call C:\ProgramData\miniconda3\Scripts\activate.bat ldo
python "%CD%"\scripts\relauncher.py

:PROMPT
set SETUPTOOLS_USE_DISTUTILS=stdlib
IF EXIST "models\ldm\stable-diffusion-v1\model.ckpt" (
  python scripts/relauncher.py
) ELSE (
  ECHO Your model file does not exist! Place it in 'models\ldm\stable-diffusion-v1' with the name 'model.ckpt'.
)
