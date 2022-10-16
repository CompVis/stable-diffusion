#! /usr/bin/env bash

# This file contains bunch of compatibility tests that ensures
# that the API interface of `scripts/legacy-api.py` remains stable

set -e

OUTDIR=$(mktemp -d)

echo "Using directory $OUTDIR"

# Start API
python -u scripts/legacy_api.py --web --host=localhost --port=3333 --outdir=$OUTDIR &> $OUTDIR/sd.log &
APP_PID=$!

echo "Wait for server to startup"

tail -f -n0 $OUTDIR/sd.log | grep -qe "Point your browser at"

echo "Started, continuing"

if [ $? == 1 ]; then
    echo "Search terminated without finding the pattern"
fi

# Generate image
RESULT=$(curl -v -X POST -d '{"index":0,"variation_amount":0,"with_variations":"","steps":25,"width":512,"seed":"1337","prompt":"A cat wearing a hat","strength":0.5,"initimg":null,"cfg_scale":2,"iterations":1,"upscale_level":0,"upscale_strength":0,"sampler_name":"k_euler","height":512}' localhost:3333 | grep result)

# Test 01 - Image contents
FILENAME=$(echo $RESULT | jq -r .url)

ACTUAL_CHECKSUM=$(sha256sum $FILENAME)
EXPECTED_CHECKSUM="a77799226a4dfc62a1674498e575c775da042959a4b90b13e26f666c302f079f"

if [ "$ACTUAL_CHECKSUM" != "$EXPECTED_CHECKSUM" ]; then
    echo "Expected hash $EXPECTED_CHECKSUM but got hash $ACTUAL_CHECKSUM"
    kill $APP_PID
    # rm -r $OUTDIR
    exit 33
fi

# Assert output

# Cleanup
kill $APP_PID
# rm -r $OUTDIR
