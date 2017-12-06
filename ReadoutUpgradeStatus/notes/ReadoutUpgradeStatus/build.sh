#!/bin/sh

BASHMODE=${1:-0}
#

rm *.aux
pdflatex status.tex
pdflatex status.tex
bibtex status.aux
bibtex status.aux
bibtex status.aux
pdflatex status.tex
pdflatex status.tex
#cp status.pdf ~/Dropbox/tmp

if [ $BASHMODE -eq 0 ]; then
    evince status.pdf
    open status.pdf
else
    echo
    echo "----------------------------------------------------------------------------"
    echo " PDF Output file: ${PWD}/status.pdf"
    echo "----------------------------------------------------------------------------"
    echo
fi
