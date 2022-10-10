#!/bin/bash

(
echo '%%%% DO NOT MODIFY: AUTOMATICALLY GENERATED.'
echo '\begin{table}'
echo '\begin{scriptsize}'
echo '\begin{minipage}{.33\textwidth}'
echo '\begin{tabular}{|c|c|c|c|}'
echo '\hline'
echo 'Problem & Train & Tool & Acc.\\'
echo '\hline'
grep '^L' archive_results.txt | sed 's/ / \& /g' | sed 's/$/\\\\/g' | egrep -v ' [24]0 ' | cut -c 3- | sed 's/_/-/g' | awk '{ if ( $1 != prev ) { print "\\hline" ; print "\\end{tabular}\\end{minipage} \\\\ \\begin{minipage}{.33\\textwidth}\\begin{tabular}{|c|c|c|c|}\\hline" } print $0; prev = $1 }'
echo '\hline'
echo '\end{tabular}\end{minipage}'
echo '\end{scriptsize}'
echo '\caption{\label{acc}Accuracies when learning high-level features: accuracies are quite poor. Majority is a simple majority vote as a baseline.}'
echo '\end{table}'
echo '\begin{table}'
echo '\begin{scriptsize}'
echo '\begin{minipage}{.33\textwidth}'
echo '\begin{tabular}{|c|c|c|c|}'
echo '\hline'
echo 'Problem & Train set & Tool & Perf\\'
echo '\hline'
grep '^O' archive_results.txt | sed 's/ / \& /g' | sed 's/$/\\\\/g' | egrep -v ' [24]0 ' | cut -c 3- | sed 's/_/-/g' | awk '{ if ( $1 != prev ) { print "\\hline" ; print "\\end{tabular}\\end{minipage} \\\\ \\begin{minipage}{.33\\textwidth}\\begin{tabular}{|c|c|c|c|}\\hline" } print $0; prev = $1 }' | sed 's/[M-j]ajority/Random/g'
echo '\hline'
echo '\end{tabular}'
echo '\end{minipage}'
echo '\end{scriptsize}'
echo '\caption{\label{opt}Frequency of success when optimizing the probability of failures $z\mapsto P(Quality(G(z))=Good)$. Results are good, in spite of the poor accuracies in Table \ref{acc}. Random is the standard Vanilla StableDiffusion reroll as a baseline.}'
echo '\end{table}'
) > ltable.tex
sed -i.tmp 's/[0-9]\.[0-9][0-9]/&PROUTPROUT/g' ltable.tex
sed -i.tp 's/PROUTPROUT[0-9]*//g' ltable.tex
open ltable.tex
