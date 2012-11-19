#!/bin/bash

export paper_root=..
cp ${paper_root}/ahnert_demidov_rupp_2012.tex darg2012.tex
cp ${paper_root}/ref.bib .
cp ${paper_root}/siam.bst .
cp ${paper_root}/siam10.clo .
cp ${paper_root}/siamltex.cls .
cp ${paper_root}/subeqn.clo .

ps2pdf ${paper_root}/data/lorenz_ensemble/perfmtx.eps lrnzperf.pdf
ps2pdf ${paper_root}/data/phase_oscillator_chain/perfmtx.eps phosperf.pdf
ps2pdf ${paper_root}/data/disordered_ham_lattice/perfmtx.eps hamlperf.pdf

ps2pdf ${paper_root}/data/lorenz_ensemble/scaling.eps lrnzscl.pdf
ps2pdf ${paper_root}/data/phase_oscillator_chain/scaling.eps phosscl.pdf
ps2pdf ${paper_root}/data/disordered_ham_lattice/scaling.eps hamlscl.pdf

sed -e 's/data\/lorenz_ensemble\/perfmtx/lrnzperf/' \
    -e 's/data\/phase_oscillator_chain\/perfmtx/phosperf/' \
    -e 's/data\/disordered_ham_lattice\/perfmtx/hamlperf/' \
    -e 's/data\/lorenz_ensemble\/scaling/lrnzscl/' \
    -e 's/data\/phase_oscillator_chain\/scaling/phosscl/' \
    -e 's/data\/disordered_ham_lattice\/scaling/hamlscl/' \
    -i darg2012.tex

texi2dvi -c -p darg2012.tex
