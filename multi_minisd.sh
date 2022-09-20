#!/bin/bash

set -e -x

touch empty_file
rm empty_file
touch empty_file
touch SD_prout_${random}.png
touch SD_prout_${random}.txt
mv SD_*.png poubelle/
mv SD_*.txt poubelle/
# Initialization: run SD and create an image, with rank 1.
touch goodbad.py
rm goodbad.py
touch goodbad.py
echo "good = []" >> goodbad.py
echo "bad = []" >> goodbad.py
python minisd.py
    #sentinel=${RANDOM}
    #touch SD_image_${sentinel}.png
    #touch SD_latent_${sentinel}.txt
mylist="`ls -ctr SD*_image_*.png | tail -n 1`"
myranks=1

for i in `seq 30`
do
    # Now an iteration.
    echo Current images = $mylist
    echo Current ranks = $myranks
    #sentinel=${RANDOM}
    #touch SD_image_${sentinel}.png
    #touch SD_latent_${sentinel}.txt
    echo GENERATING FOUR IMAGES.
    python minisd.py
    python minisd.py
    python minisd.py
    python minisd.py
    for img in `ls -ctr SD*_image_*.png | tail -n 4`
    do
        montage $mylist $img -mode Concatenate -tile 5x output.png
        open --wait output.png  
        read -p "Rank of the last image ?" rank
        mylist="$mylist $img`
        mynewranks=""
        for r in $myranks
        do
            [[ $r -ge $rank ]] && r=$(( $r + 1 ))
            mynewranks="$mynewranks $r"
        done
        myranks="$mynewranks $rank"
        #echo Before sorting ===========================
        #echo $myranks
        #echo $mylist
        #sleep 2
    
        # Now sorting
        mynewlist=""
        mynewranks=""
        touch goodbad.py
        rm goodbad.py
        touch goodbad.py
        echo "good = []" >> goodbad.py
        echo "bad = []" >> goodbad.py
    
        for r in `seq 20`
        do
          for k in `seq 20`
          do
             [[ `echo $myranks | cut -d ' ' -f $k` -eq $r ]] && echo "FOUND $k for $r!"
             my_image="`echo $mylist | cut -d ' ' -f $k`"
             [[ `echo $myranks | cut -d ' ' -f $k` -eq $r ]] && mynewranks="$mynewranks `echo $myranks | cut -d ' ' -f $k`"
             [[ `echo $myranks | cut -d ' ' -f $k` -eq $r ]] && mynewlist="$mynewlist `echo $mylist | cut -d ' ' -f $k`"
             if [[ `echo $myranks | cut -d ' ' -f $k` -eq $r ]]
             then 
                echo Found $my_image at rank $k for $r
                if [[ $r -le 5 ]]
                then
                cat empty_file `echo $my_image | sed 's/image_[0-9]*.png/PROUTPROUT&/g' | sed 's/PROUTPROUTimage/latent/g' | sed 's/\.png/\.txt/g'` | sed "s/.*/good += [&]/g"  >> goodbad.py
                else
                cat empty_file `echo $my_image | sed 's/image_[0-9]*.png/PROUTPROUT&/g' | sed 's/PROUTPROUTimage/latent/g' | sed 's/\.png/\.txt/g'` | sed "s/.*/bad += [&]/g" >> goodbad.py
                fi
             fi
             echo "" >> goodbad.py
          done
        done
    done
    myranks=$mynewranks
    mylist=$mynewlist
    echo After sorting ===========================
    echo $myranks
    echo $mylist
    #sleep 2
    
done


