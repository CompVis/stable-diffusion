#!/bin/bash
set -e
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda activate ldm
exec "$@"
