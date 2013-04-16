cmake_minimum_required(VERSION 2.8)
project(gpgpu_with_modern_cpp)


find_package(LATEX)

if (LATEX_COMPILER)
    include(UseLATEX.cmake)

    add_latex_document(gpgpu_with_modern_cpp.tex
	IMAGES data/lorenz_ensemble/perfmtx.eps
	       data/lorenz_ensemble/scaling.eps
	       data/phase_oscillator_chain/perfmtx.eps
	       data/phase_oscillator_chain/scaling.eps
	       data/disordered_ham_lattice/perfmtx.eps
	       data/disordered_ham_lattice/scaling.eps
	BIBFILES ref.bib
	DEPENDS siamltex.cls siam.bst siam10.clo subeqn.clo
	DEFAULT_PDF
	MANGLE_TARGET_NAMES
	)

    add_latex_document(cover_letter.tex   DEFAULT_PDF MANGLE_TARGET_NAMES)
    add_latex_document(cover_revision.tex DEFAULT_PDF MANGLE_TARGET_NAMES)
    add_latex_document(cover_final.tex    DEFAULT_PDF MANGLE_TARGET_NAMES)
endif (LATEX_COMPILER)

add_subdirectory(src)
