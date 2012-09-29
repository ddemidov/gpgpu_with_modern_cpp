cmake_minimum_required(VERSION 2.8)
project(vexcl_odeint_paper_sources)

find_package(Boost)

find_package(CUDA QUIET)

if (CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "-arch=sm_13")
endif (CUDA_FOUND)

set(ODEINT_ROOT   "/home/demidov/work/karsten/odeint-v2"     CACHE STRING "odeint root")
set(VEXCL_ROOT    "/home/demidov/work/vexcl"                 CACHE STRING "VexCL root")
set(VIENNACL_ROOT "/home/demidov/work/opencl/ViennaCL-1.4.0-preview" CACHE STRING "ViennaCL root")

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
