cmake_minimum_required(VERSION 2.8)

project(disordered_ham_lattice)

find_package(Boost)

set(ODEINT_ROOT   "/home/karsten/src/odeint-v2")
set(VEXCL_ROOT    "/home/karsten/boost/vexcl")
set(VIENNACL_ROOT "/home/karsten/boost/testing/ViennaCL-1.3.1")
set(CUDA_ROOT     "/usr/local/cuda/include")

include_directories(
    ${Boost_INCLUDE_DIRS}
    ${ODEINT_ROOT}
    ${VEXCL_ROOT}
    ${CUDA_ROOT}
    ${VIENNACL_ROOT}
    )

add_executable(vexcl_disordered_lattice vexcl_disordered_lattice.cpp)
target_link_libraries(vexcl_disordered_lattice OpenCL)
set_target_properties(vexcl_disordered_lattice PROPERTIES COMPILE_FLAGS -std=c++0x)

# add_executable(viennacl_oscillator viennacl_oscillator.cpp)
# target_link_libraries(viennacl_oscillator OpenCL)
# set_target_properties(viennacl_oscillator PROPERTIES COMPILE_FLAGS -std=c++0x)

# foreach(script run_on_cpu run_on_1gpu run_on_2gpu run_on_3gpu)
#    configure_file(
#	${CMAKE_CURRENT_SOURCE_DIR}/${script}.sge
#	${CMAKE_CURRENT_BINARY_DIR}/${script}.sge
#	COPYONLY
#	)
#endforeach(script)
