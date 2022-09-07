#!/bin/bash

for i in `seq 10 20`; do
python scripts/img2img.py \
    --seed1 $i \
    --seed2 $((i + 100)) \
    --init-img "cathedral.png" \
    --strength 0.999 \
    --prompt1 "endless book stairway to universe, sky full of clouds, art by greg rutkowski and peter mohrbacher, featured in artstation, octane render, cinematic, elegant, intricate, ultra detailed, rule of thirds, professional lighting, unreal engine, fantasy, concept art, sharp focus, illustration, 8 k " \
    --prompt2 "endless book stairway to universe, sky full of clouds, art by greg rutkowski and peter mohrbacher, featured in artstation, octane render, cinematic, elegant, intricate, ultra detailed, rule of thirds, professional lighting, unreal engine, fantasy, concept art, sharp focus, illustration, 8 k " \
    --n_iter 200 \
    --output_video stairwayvlouds-$i.mp4
done
