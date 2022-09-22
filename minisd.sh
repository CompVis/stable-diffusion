#!/bin/bash
if compgen -G inoculations/SD*.png > /dev/null ; then
file=`ls -ctr inoculations/SD*.png | tail -n 1`
echo Transformation:
latent=`echo $file | sed 's/image_[0-9]*.png/PROUTPROUT&/g' | sed 's/PROUTPROUTimage/latent/g' | sed 's/\.png/\.txt/g'`
echo Image=$file
echo Latent=$latent
mv $file $latent .
else
python minisd.py
fi 
