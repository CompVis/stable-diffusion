tar -zcvf ~/bigpack2.tgz `ls -ctrl | grep 'Sep.16' | grep '\.png' | sed 's/.* //g'` | wc -l
ls -ctlr ~/bigpack2.tgz
