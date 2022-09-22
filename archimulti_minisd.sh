#!/bin/bash

for u in `seq 25`
do
export prompt="A photographic portrait of a young woman, tilted head, from below, red hair, green eyes, smiling."
python minisd.py
export prompt="A photo of a cute armoured bloody red panda fighting off tentacles with daggers."
python minisd.py
export prompt="A photo of a woman fighting off tentacles with guns."
python minisd.py
export prompt="A cute armoured red panda fighting off zombies with karate."
python minisd.py
export prompt="An armored Mark Zuckerberg fighting off bloody tentacles in the jungle."
python minisd.py
done
