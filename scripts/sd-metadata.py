#!/usr/bin/env python

import sys
import json
from ldm.invoke.pngwriter import retrieve_metadata

if len(sys.argv) < 2:
    print("Usage: file2prompt.py <file1.png> <file2.png> <file3.png>...")
    print("This script opens up the indicated invoke.py-generated PNG file(s) and prints out their metadata.")
    exit(-1)

filenames = sys.argv[1:]
for f in filenames:
    try:
        metadata = retrieve_metadata(f)
        print(f'{f}:\n',json.dumps(metadata['sd-metadata'], indent=4))
    except FileNotFoundError:
        sys.stderr.write(f'{f} not found\n')
        continue
    except PermissionError:
        sys.stderr.write(f'{f} could not be opened due to inadequate permissions\n')
        continue
