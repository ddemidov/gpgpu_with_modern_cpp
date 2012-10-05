cmake_minimum_required(VERSION 2.8)
project(vexcl_odeint_paper_sources)

# set( Boost_NO_SYSTEM_PATHS 1 )
find_package(Boost)

find_package(CUDA QUIET)

if (CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "-arch=sm_13")
endif (CUDA_FOUND)

set(ODEINT_ROOT    $ENV{ODEINT_ROOT}   CACHE STRING "odeint root")
set(VEXCL_ROOT     $ENV{VEXCL_ROOT}    CACHE STRING "VexCL root")
set(VIENNACL_ROOT  $ENV{VIENNACL_ROOT} CACHE STRING "ViennaCL root")

MESSAGE(STATUS "odeint found in ${ODEINT_ROOT}")
MESSAGE(STATUS "VexCL found in ${VEXCL_ROOT}")
MESSAGE(STATUS "ViennaCL found in ${VIENNACL_ROOT}")
MESSAGE(STATUS "Boost found in ${BOOST_ROOT}")

include_directories(
    ${BOOST_ROOT}
    ${ODEINT_ROOT}
    ${VEXCL_ROOT}
    ${VIENNACL_ROOT}
    )

add_subdirectory(damped_oscillator)
add_subdirectory(disordered_ham_lattice)
add_subdirectory(lorenz_ensemble)
add_subdirectory(phase_oscillator_chain)


configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh
    ${CMAKE_CURRENT_BINARY_DIR}/run_tests.sh
    COPYONLY
    )

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/run_tahiti_tests.sh
    ${CMAKE_CURRENT_BINARY_DIR}/run_tahiti_tests.sh
    COPYONLY
    )

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/gather_data.sh
    ${CMAKE_CURRENT_BINARY_DIR}/gather_data.sh
    COPYONLY
    )
