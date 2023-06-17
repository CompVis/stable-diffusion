@echo off

echo Loading Python and Anaconda environment, this may take up to a minute...

set conda_env_name=lda

set paths="%ProgramData%\miniconda3"
set paths=%paths%;"%USERPROFILE%\miniconda3"
set paths=%paths%;"%ProgramData%\anaconda3"
set paths=%paths%;"%USERPROFILE%\anaconda3"

set custom_conda_path=%1

shift
set python=%1
:loop
shift
if [%1]==[] goto afterloop
set python=%python% %1
goto loop
:afterloop

IF NOT %custom_conda_path%=="Select Folder" (
  set paths=%custom_conda_path%
)

for %%a in (%paths%) do ( 
 if EXIST "%%a\Scripts\activate.bat" (
    SET CONDA_PATH=%%a
    ::echo anaconda3/miniconda3 detected in %%a
    goto :foundPath
 )
)

IF "%CONDA_PATH%"=="" (
  IF NOT %custom_conda_path%=="Select Folder" (
    call color 04 && echo Anaconda3/Miniconda3 not found in custom path: %custom_conda_path%. Please check your settings
  ) else (
    call color 04 && echo Anaconda3/Miniconda3 not found. Please install from here https://docs.conda.io/en/latest/miniconda.html
  )
  pause
  exit
)

:foundPath
call %CONDA_PATH%\Scripts\activate.bat
call %CONDA_PATH%\Scripts\activate.bat "%conda_env_name%"
call python scripts\%python% || color 04 && echo An error has occured. && conda deactivate && timeout /t -1 && exit
