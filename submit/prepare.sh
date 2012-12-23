#!/bin/bash

export paper_root=..

cp ${paper_root}/gpgpu_with_modern_cpp.tex gpgpucpp.tex
cp ${paper_root}/ref.bib .
cp ${paper_root}/siam.bst .
cp ${paper_root}/siam10.clo .
cp ${paper_root}/siamltex.cls .
cp ${paper_root}/subeqn.clo .

cp ${paper_root}/data/lorenz_ensemble/perfmtx.eps        lrnzperf.eps
cp ${paper_root}/data/phase_oscillator_chain/perfmtx.eps phosperf.eps
cp ${paper_root}/data/disordered_ham_lattice/perfmtx.eps hamlperf.eps
cp ${paper_root}/data/lorenz_ensemble/scaling.eps        lrnzscl.eps
cp ${paper_root}/data/phase_oscillator_chain/scaling.eps phosscl.eps
cp ${paper_root}/data/disordered_ham_lattice/scaling.eps hamlscl.eps

for fig in lrnzperf phosperf hamlperf lrnzscl phosscl hamlscl; do
    ps2pdf14 -dEPSCrop -dMaxSubsetPct=100 -dCompatibilityLevel=1.3 \
        -dSubsetFonts=true -dEmbedAllFonts=true -dAutoFilterColorImages=false \
        -dAutoFilterGrayImages=false -dColorImageFilter=/FlateEncode \
        -dGrayImageFilter=/FlateEncode -dMonoImageFilter=/FlateEncode \
        ${fig}.eps ${fig}.pdf
done


sed -e 's/data\/lorenz_ensemble\/perfmtx/lrnzperf/' \
    -e 's/data\/phase_oscillator_chain\/perfmtx/phosperf/' \
    -e 's/data\/disordered_ham_lattice\/perfmtx/hamlperf/' \
    -e 's/data\/lorenz_ensemble\/scaling/lrnzscl/' \
    -e 's/data\/phase_oscillator_chain\/scaling/phosscl/' \
    -e 's/data\/disordered_ham_lattice\/scaling/hamlscl/' \
    -i gpgpucpp.tex

texi2dvi -c -p gpgpucpp.tex
