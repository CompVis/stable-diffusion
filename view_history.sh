#montage `ls -ctr SD*imag*.png | head -n 15 | tail -n 14` -mode concatenate -tile 7x zuck1.png
#montage `ls -ctr SD*imag*.png | head -n 29 | tail -n 14` -mode concatenate -tile 7x zuck2.png
montage `ls -ctr SD*imag*.png | tail -n 28` -mode concatenate -tile 7x history.png
open history.png
#cp history.png zuck3.png
