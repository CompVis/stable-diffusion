#!/bin/bash
#montage `ls -ctr SD*imag*.png | head -n 15 | tail -n 14` -mode concatenate -tile 7x zuck1.png
#montage `ls -ctr SD*imag*.png | head -n 29 | tail -n 14` -mode concatenate -tile 7x zuck2.png
#montage `ls -ctr SD*imag*.png | tail -n 28` -mode concatenate -tile 7x history.png
#montage $( ls *`ls -ctr SD*.png | sed 's/.*image_//g' | tail -n 1 | sed 's/_.*//g'`*.png | sort | tail -n 60 | sort )  -mode concatenate -tile 5x history.png
#montage $( ls *`ls -ctr SD*.png | sed 's/.*image_//g' | tail -n 1 | sed 's/_.*//g'`_0_11.png | sort  ) $( ls *`ls -ctr SD*.png | sed 's/.*image_//g' | tail -n 1 | sed 's/_.*//g'`_0_4.png | sort  ) $( ls *`ls -ctr SD*.png | sed 's/.*image_//g' | tail -n 1 | sed 's/_.*//g'`_1_?.png | sort -n )  $( ls *`ls -ctr SD*.png | sed 's/.*image_//g' | tail -n 1 | sed 's/_.*//g'`_1_??.png | sort -n )  -mode concatenate -tile 5x history.png
#montage $( ls SD*`ls -ctr SD*.png | sed 's/.*image_//g' | tail -n 1 | sed 's/_.*//g'`*.png | sort  )  -mode concatenate -tile 7x history.png
#open history.png
open $( ls *`ls -ctr SD*.png | sed 's/.*image_//g' | tail -n 1 | sed 's/_.*//g'`*.png | tail -n 5 | sort  ) 
#cp history.png zuck3.png
