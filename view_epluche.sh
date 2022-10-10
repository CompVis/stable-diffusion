#!/bin/bash

(
echo '%%%% DO NOT MODIFY: AUTOMATICALLY GENERATED.'
echo '\begin{table}'
echo '\begin{tabular}{|c|c|c|c|}'
echo '\hline'
echo 'Problem & Training set & Tool & Accuracy \\'
echo '\hline'
grep '^L' archive_results.txt | grep Decisi | sed 's/ / \& /g' | sed 's/$/\\\\/g' | cut -c 3- | sed 's/_/-/g'
echo '\hline'
echo '\end{tabular}'
echo '\caption{\label{acc}Accuracies when learning high-level features: accuracies are quite poor.}'
echo '\end{table}'
echo '\begin{table}'
echo '\begin{tabular}{|c|c|c|c|}'
echo '\hline'
echo 'Problem & Training set & Tool & Generation perf\\'
echo '\hline'
grep '^O' archive_results.txt | grep Decisi | sed 's/ / \& /g' | sed 's/$/\\\\/g' | cut -c 3- | sed 's/_/-/g'
echo '\hline'
echo '\caption{\label{opt}Frequency of success when optimizing the probability of failures $z\mapsto P(Quality(G(z))=Good)$. Results are good, in spite of the poor accuracies in Table \ref{app}.}'
echo '\end{tabular}'
echo '\end{table}'
) > ltable.tex
open ltable.tex
