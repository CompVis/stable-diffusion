#!/bin/bash
while [ $# -gt 0 ]; do
    if [[ $1 == "-"* ]]; then
        v="${1/-/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

# -p : name cluster
# -g : nb gpu
# -w : walltime
# -b : batchsize
# -r : checkpoint file or dir

if [ -z "$w" ];
then
    wtime=60
else
    wtime=$w
fi

gpus=$(($g-1))
gseq=$(seq -s ',' $gpus)

echo -p $p
echo -g $g
echo -gseq $gseq
echo -w $w
echo -b $b
echo -r "'$r'"

if [ -z "$p" ];
then
    if [ -z "$r" ];
    then
        oarsub -q production -l host=1/gpu=$g,walltime=$wtime --notify mail:cyril.regan@loria.fr "cd ~/GENS/FORK_stable-diffusion ; $(which singularity) run --nv ../stable_lning.sif /conda/bin/conda run -n ldm --no-capture-output python main.py --base configs/autoencoder/vqgan.yaml -t --gpus $gseq --batch_size $b ; sleep infinity"
    else
        oarsub -q production -l host=1/gpu=$g,walltime=$wtime --notify mail:cyril.regan@loria.fr "cd ~/GENS/FORK_stable-diffusion ; $(which singularity) run --nv ../stable_lning.sif /conda/bin/conda run -n ldm --no-capture-output python main.py --base configs/autoencoder/vqgan.yaml -t --gpus $gseq --batch_size $b --resume '$r' ; sleep infinity"
    fi
else
    if [ -z "$r" ];
    then
        oarsub -q production -p $p -l host=1/gpu=$g,walltime=$wtime --notify mail:cyril.regan@loria.fr "cd ~/GENS/FORK_stable-diffusion ; $(which singularity) run --nv ../stable_lning.sif /conda/bin/conda run -n ldm --no-capture-output python main.py --base configs/autoencoder/vqgan.yaml -t --gpus $gseq --batch_size $b ; sleep infinity"
    else
        oarsub -q production -p $p -l host=1/gpu=$g,walltime=$wtime --notify mail:cyril.regan@loria.fr "cd ~/GENS/FORK_stable-diffusion ; $(which singularity) run --nv ../stable_lning.sif /conda/bin/conda run -n ldm --no-capture-output python main.py --base configs/autoencoder/vqgan.yaml -t --gpus $gseq --batch_size $b --resume '$r' ; sleep infinity"
    fi
fi
