cmake_minimum_required(VERSION 2.8)
project(ahnert_demidov_2012)


find_package(LATEX)

if (LATEX_COMPILER)
    include(UseLATEX.cmake)

    add_latex_document(ahnert_demidov_rupp_2012.tex
	IMAGES data/lorenz_ensemble/perfcmp.eps
	       data/lorenz_ensemble/perfcmp_tesla.eps
	       data/lorenz_ensemble/perfcmp_tahiti.eps
	       data/lorenz_ensemble/scaling.eps
	       data/phase_oscillator_chain/perfcmp.eps
	       data/phase_oscillator_chain/perfcmp_tesla.eps
	       data/phase_oscillator_chain/perfcmp_tahiti.eps
	       data/phase_oscillator_chain/scaling.eps
	       data/damped_oscillator/perfcmp.eps
	       data/damped_oscillator/perfcmp_tesla.eps
	       data/damped_oscillator/perfcmp_tahiti.eps
	       data/damped_oscillator/scaling.eps
	       data/damped_oscillator/genkernel.eps
	       data/disordered_ham_lattice/perfcmp.eps
	       data/disordered_ham_lattice/perfcmp_tesla.eps
	       data/disordered_ham_lattice/perfcmp_tahiti.eps
	       data/disordered_ham_lattice/scaling.eps
	BIBFILES ref.bib
	DEPENDS siamltex.cls siam.bst siam10.clo subeqn.clo
	DEFAULT_PDF
	MANGLE_TARGET_NAMES
	)
endif (LATEX_COMPILER)

add_subdirectory(src)
