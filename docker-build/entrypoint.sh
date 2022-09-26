#!/bin/bash

cd /stable-diffusion

if [ $# -eq 0 ]; then
    python3 scripts/dream.py --full_precision -o /data
    # bash
else
    python3 scripts/dream.py --full_precision -o /data "$@"
fi