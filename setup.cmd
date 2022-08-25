@echo off
CD /d %~dp0
conda env create -f environment.yaml

