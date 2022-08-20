@echo off
IF NOT EXIST CONDA umamba create -r conda -f environment.yaml -y
call conda\condabin\activate.bat ldm
cls

:PROMPT
python scripts/txt2img_gradio.py