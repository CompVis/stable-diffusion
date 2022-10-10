#!/bin/bash
for u in `ls toto*.txt | grep "${1:-.}" | gshuf`
do
    echo $u
    v=`echo $u | sed 's/toto//g' | sed 's/_.*//g'`
    touch goodbad.py
    rm goodbad.py
    touch goodbad.py
    echo "pb=\"$v\"" >> goodbad.py
    echo "good = []" >> goodbad.py
    echo "bad = []" >> goodbad.py
    for v in `cat $u | grep R$ | awk '{print $1}'`
    do
        echo "bad += [`cat ${v}*.txt`]" >> goodbad.py
    done
    for v in `cat $u | grep L$ | awk '{print $1}'`
    do
        echo "good += [`cat ${v}*.txt`]" >> goodbad.py
    done
    echo "print(len(good))" >> goodbad.py
    echo "print(len(bad))" >> goodbad.py
    cat learn.py >> goodbad.py
    python goodbad.py
    mv goodbad.py goodbad_${v}.py
done
