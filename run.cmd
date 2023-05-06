@echo off

echo Loading Python and Anaconda environment, this may take up to a minute...

set conda_env_name=lda

set paths="%ProgramData%\miniconda3"
set paths=%paths%;"%USERPROFILE%\miniconda3"
set paths=%paths%;"%ProgramData%\anaconda3"
set paths=%paths%;"%USERPROFILE%\anaconda3"

for %%a in (%paths%) do ( 
 if EXIST "%%a\Scripts\activate.bat" (
    SET CONDA_PATH=%%a
    ::echo anaconda3/miniconda3 detected in %%a
    goto :foundPath
 )
)

IF "%CONDA_PATH%"=="" (
  echo anaconda3/miniconda3 not found. Install from here https://docs.conda.io/en/latest/miniconda.html
  exit /b 1
)

:foundPath
call %CONDA_PATH%\Scripts\activate.bat
call %CONDA_PATH%\Scripts\activate.bat "%conda_env_name%"
call python scripts\%* || color 04 && echo An error has occured. && conda deactivate && timeout /t -1 && exit
