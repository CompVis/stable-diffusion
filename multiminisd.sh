#!/bin/bash
touch SD.prout.${RANDOM}
mv SD*.* poubelle/

numimages=12
for m in 5   #2 5 3 4 1
do
export mu=$m
for d in 1 0.5 0
do
export decay=$d
for ngo in OnePlusOne DiscreteOnePlusOne RandomSearch DiscreteLenglerOnePlusOne 
do
export ngoptim=$ngo
for sl in tree nn logit 
do
export skl=$sl
for es in False True
do
export earlystop=$es
for eps in 0.0001
do
export epsilon=$eps

export prompt="A close up photographic portrait of a young woman with uniformly colored hair."
directory=biased_${epsilon}_rw_experiment${numimages}_images_${mu}_${ngoptim}_${earlystop}_${skl}_${decay}
mkdir $directory
for u in `seq $numimages`
do
cp goodbad_learnbluehair.py goodbad.py
python minisd.py
./view_history.sh
sleep 1
done
cp history.png SD* *.py *.sh $directory

mv SD*.* poubelle/
done
done
done
done
done
done
