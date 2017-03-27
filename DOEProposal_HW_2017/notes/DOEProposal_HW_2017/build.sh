
BATCHMODE=1
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

if [ $BATCHMODE -eq 1 ]; then
    echo
    echo "----------------------------------------------------------------------------"
    echo " PDF Output file: ${PWD}/proposal.pdf"
    echo "----------------------------------------------------------------------------"
    echo
else
    evince proposal.pdf
    open proposal.pdf
fi
