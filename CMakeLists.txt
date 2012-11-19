cmake_minimum_required(VERSION 2.8)
project(ahnert_demidov_2012)


find_package(LATEX)

if (LATEX_COMPILER)
    include(UseLATEX.cmake)

    add_latex_document(ahnert_demidov_rupp_2012.tex
	IMAGES data/lorenz_ensemble/perfmtx.eps
	       data/lorenz_ensemble/scaling.eps
	       data/phase_oscillator_chain/perfmtx.eps
	       data/phase_oscillator_chain/scaling.eps
	       data/damped_oscillator/perfmtx.eps
	       data/damped_oscillator/scaling.eps
	       data/disordered_ham_lattice/perfmtx.eps
	       data/disordered_ham_lattice/scaling.eps
	BIBFILES ref.bib
	DEPENDS siamltex.cls siam.bst siam10.clo subeqn.clo
	DEFAULT_PDF
	MANGLE_TARGET_NAMES
	)

    add_latex_document(cover_letter.tex DEFAULT_PDF MANGLE_TARGET_NAMES)
endif (LATEX_COMPILER)

add_subdirectory(src)
