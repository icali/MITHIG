#!/bin/sh

BASHMODE=${1:-0}
#

rm *.aux
pdflatex proposal.tex
pdflatex proposal.tex
bibtex proposal.aux
bibtex proposal.aux
bibtex proposal.aux
pdflatex proposal.tex
pdflatex proposal.tex
#cp proposal.pdf ~/Dropbox/tmp

if [ $BASHMODE -eq 0 ]; then
    evince proposal.pdf
    open proposal.pdf
else
    echo
    echo "----------------------------------------------------------------------------"
    echo " PDF Output file: ${PWD}/proposal.pdf"
    echo "----------------------------------------------------------------------------"
    echo
fi
