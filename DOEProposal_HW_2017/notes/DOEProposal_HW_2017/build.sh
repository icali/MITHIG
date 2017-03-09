rm *.aux
pdflatex proposal.tex
pdflatex proposal.tex
bibtex proposal.aux
bibtex proposal.aux
bibtex proposal.aux
pdflatex proposal.tex
pdflatex proposal.tex
#cp proposal.pdf ~/Dropbox/tmp
evince proposal.pdf
open proposal.pdf