#!/bin/bash

echo Parametrization and initialization.
#export prompt="A close up photographic portrait of a young woman with uniformly colored hair."
export prompt="An armored Mark Zuckerberg fighting off bloody tentacles in the jungle."
lambda=18
cp basic_inoculation_uniformly/SD*.* inoculations/ 


set -e

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
#python minisd.py
./minisd.sh
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
    echo "GENERATING $lambda IMAGES ================================"
    cat goodbad.py | awk '!x[$0]++' > goodbad2.py
    mv goodbad2.py goodbad.py
    echo "`grep -c 'good +=' goodbad.py` positive examples"
    echo "`grep -c 'bad +=' goodbad.py` negative examples"
    for kk in `seq $lambda`
    do
      echo "generating image $kk / $lambda"
      #python minisd.py
      ./minisd.sh
    done
    list_of_four_images="`ls -ctr SD*_image_*.png | tail -n $lambda`"
#    my_new_list=""
#    my_new_ranks=""
#    # We stop at 19 so that it becomes 20 with the new one
#    for k in `seq 19`
#    do
#       my_new_list="$my_new_list `echo $mylist | cut -d ' ' -f $k`"
#       my_new_ranks="$my_new_ranks `echo $myranks | cut -d ' ' -f $k`"
#    done
#    mylist=`echo $my_new_list | sed 's/[ ]*$//g'`
#    myranks=`echo $my_new_ranks | sed 's/[ ]*$//g'`
#    echo "After limiting to 19, we get $mylist and $myranks "
    for img in $list_of_four_images
    do
        echo We add image $img =======================
        montage $mylist $img -mode Concatenate -tile 5x output.png
        open --wait output.png  
        # read -t 1 prout
        read -p "Rank of the last image ?" rank
        echo "Provided rank: $rank"
        mylist="$mylist $img"
        if [[ $rank -le 0 ]]
        then
        read -p "Enter all ranks !!!!" myranks
        else 
        mynewranks=""
        for r in $myranks
        do
            [[ $r -ge $rank ]] && r=$(( $r + 1 ))
            mynewranks="$mynewranks $r"
        done
        myranks="$mynewranks $rank"
        fi
        #echo Before sorting ===========================
        #echo $myranks
        #echo $mylist
        #sleep 5
    
        # Now sorting
        mynewlist=""
        mynewranks=""
        sed -i.backup 's/good +=.*//g' goodbad.py
        num_good=`cat goodbad.py | grep 'good +=' | wc -l `
        echo "Num goods in file: $num_good"
        num_good=$(( $num_good / 2 + 5 ))
        echo "Num goods after update: $num_good"
        echo "We keep the $num_good best."
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
                if [[ $r -le $num_good ]]
                then
                cat empty_file `echo $my_image | sed 's/image_[0-9]*.png/PROUTPROUT&/g' | sed 's/PROUTPROUTimage/latent/g' | sed 's/\.png/\.txt/g'` | sed "s/.*/good += [&]/g"  >> goodbad.py
                else
                cat empty_file `echo $my_image | sed 's/image_[0-9]*.png/PROUTPROUT&/g' | sed 's/PROUTPROUTimage/latent/g' | sed 's/\.png/\.txt/g'` | sed "s/.*/bad += [&]/g" >> goodbad.py
                fi
                echo "" >> goodbad.py
                break
             fi
          done
        done
        myranks=$mynewranks
        mylist=$mynewlist
    done
    echo After sorting ===========================
    echo $myranks
    echo $mylist
    #sleep 2
    
done


