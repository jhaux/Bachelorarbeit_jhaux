#
#  Makefile: Bachelor Thesis
#
#  Author: Johannes Haux
#

DOCUNAME = bachelor_thesis
BEAMERNAME = bachelor_presentation
LATEXCC = pdflatex  # used for thesis
LATEXBB = xelatex   # used for beamer presentation
PYTHON = python


.PHONY: clean
clean:
	rm -f *.{log,aux,pdf}
	rm -f ./plot/pic/*

.PHONY: docu
docu:
	$(LATEXCC) ./docu/$(DOCUNAME).tex
	$(LATEXCC) ./docu/$(DOCUNAME).tex
	$(LATEXCC) ./docu/$(DOCUNAME).tex

.PHONY: pres
pres:
	$(LATEXBB) ./pres/$(BEAMERNAME).tex
	$(LATEXBB) ./pres/$(BEAMERNAME).tex
	$(LATEXBB) ./pres/$(BEAMERNAME).tex

.PHONY: plot
plot:
	$(PYTHON) plot_*.py
