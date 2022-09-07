#!/bin/bash

for i in `seq 10 20`; do
python scripts/img2img.py \
    --seed1 $i \
    --seed2 $((i + 100)) \
    --init-img "cathedral.png" \
    --strength 0.999 \
    --prompt1 "a beautiful print on paper, 8 k, frostbite 3 engine, cryengine, dof, trending on artstation, art by robert gibbings 1 9 3 4, crepuscular ray" \
    --prompt2 "a beautiful print on paper, 8 k, frostbite 3 engine, cryengine, dof, trending on artstation, art by robert gibbings 1 9 3 4, crepuscular ray" \
    --n_iter 200 \
    --output_video winter-$i.mp4
done
