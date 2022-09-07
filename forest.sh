#!/bin/bash

for i in `seq 100`; do
python scripts/img2img.py \
    --seed1 $i \
    --seed2 $((i + 100)) \
    --init-img "duckrabbit-portrait.jpg" \
    --strength 0.999 \
    --prompt1 "forest clearing landscape, studio ghibli, pixar and disney animation, sharp, rendered in unreal engine 5, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, wide angle, artbook, wallpaper, splash art, promo art, dramatic lighting, art by artgerm and greg rutkowski and bo chen and jin xiaodi" \
    --n_iter 100 \
    --output_video forest_$i.mp4
done
