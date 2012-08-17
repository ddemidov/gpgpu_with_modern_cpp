cmake_minimum_required(VERSION 2.8)
project(vexcl_odeint)
include(UseLATEX.cmake)

add_latex_document(vexcl_odeint.tex 
    BIBFILES ref.bib
    DEPENDS elsarticle.cls
    DEFAULT_PDF
    MANGLE_TARGET_NAMES
    )
