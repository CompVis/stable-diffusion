#!/bin/bash
#montage `ls -ctr SD*imag*.png | head -n 15 | tail -n 14` -mode concatenate -tile 7x zuck1.png
#montage `ls -ctr SD*imag*.png | head -n 29 | tail -n 14` -mode concatenate -tile 7x zuck2.png
#montage `ls -ctr SD*imag*.png | tail -n 28` -mode concatenate -tile 7x history.png
montage $( ls *`ls -ctr SD*.png | sed 's/.*image_//g' | tail -n 1 | sed 's/_.*//g'`*.png | sort  )  -mode concatenate -tile 9x history.png
#montage $( ls *`ls -ctr SD*.png | sed 's/.*image_//g' | tail -n 1 | sed 's/_.*//g'`*.png | sort | tail -n 60 | sort )  -mode concatenate -tile 5x history.png
open history.png
#cp history.png zuck3.png
