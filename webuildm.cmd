@echo off
call C:\ProgramData\miniconda3\Scripts\activate.bat
call conda env create -f environment.yaml
call C:\ProgramData\miniconda3\Scripts\activate.bat ldm
python "%CD%"\scripts\webui.py

:PROMPT
python scripts/webui.py