#!/usr/bin/env python3
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

import sys
import os.path

script_path = sys.argv[0]
script_args = sys.argv[1:]
script_dir,script_name = os.path.split(script_path)
script_dest = os.path.join(script_dir,'invoke.py')
os.execlp('python3','python3',script_dest,*script_args)

