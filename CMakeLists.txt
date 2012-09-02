cmake_minimum_required(VERSION 2.8)
project(ahnert_demidov_2012)

find_package(LATEX)

if (LATEX_COMPILER)
    include(UseLATEX.cmake)

    add_latex_document(ahnert_demidov_2012.tex
	IMAGES data/lorenz_ensemble/lorenz_ensemble.eps
	       data/phase_oscillator_chain/phase_oscillator.eps
	       data/damped_oscillator/damped_oscillator.eps
	       data/disordered_lattice/disordered_lattice.eps
	BIBFILES ref.bib
	DEPENDS elsarticle.cls
	DEFAULT_PDF
	MANGLE_TARGET_NAMES
	)
endif (LATEX_COMPILER)

add_subdirectory(src)
