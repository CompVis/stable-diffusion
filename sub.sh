#!/bin/bash
while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

cd ..

echo "\p: '$p'"


# oarsub -q production -p grat -l host=1/gpu=8,walltime=60 --notify mail:cyril.regan@loria.fr "cd ~/GENS/FORK_stable-diffusion ; $(which singularity) run --nv ../stable_lning.sif /conda/bin/conda run -n ldm --no-capture-output python main.py --base configs/autoencoder/vqgan.yaml -t --gpus 0,1,2,3,4,5,6,7 --batch_size 32 --resume '../RESULTS/vqgan_3000/' ; sleep infinity"
