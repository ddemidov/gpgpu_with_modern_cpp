cmake_minimum_required(VERSION 2.8)
project(ahnert_demidov_2012)

find_package(LATEX)

if (LATEX_COMPILER)
    include(UseLATEX.cmake)

    add_latex_document(ahnert_demidov_rupp_2012.tex
	IMAGES data/lorenz_ensemble/perfcmp.eps
	       data/lorenz_ensemble/scaling.eps
	       data/phase_oscillator_chain/perfcmp.eps
	       data/phase_oscillator_chain/scaling.eps
	       data/damped_oscillator/perfcmp.eps
	       data/damped_oscillator/scaling.eps
	       data/disordered_ham_lattice/perfcmp.eps
	       data/disordered_ham_lattice/scaling.eps
	BIBFILES ref.bib
	DEPENDS elsarticle.cls
	DEFAULT_PDF
	MANGLE_TARGET_NAMES
	)
endif (LATEX_COMPILER)

add_subdirectory(src)
