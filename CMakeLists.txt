cmake_minimum_required(VERSION 2.8)
project(ahnert_demidov_2012)
include(UseLATEX.cmake)

add_latex_document(ahnert_demidov_2012.tex
    BIBFILES ref.bib
    DEPENDS elsarticle.cls
    DEFAULT_PDF
    MANGLE_TARGET_NAMES
    )
