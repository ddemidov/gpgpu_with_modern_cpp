cmake_minimum_required(VERSION 2.8)
project(disordered_ham_lattice)

if (CUDA_FOUND)
    cuda_add_executable(thrust_disordered_lattice thrust_disordered_lattice.cu)
    target_link_libraries(thrust_disordered_lattice ${CUDA_cusparse_LIBRARY})

    cuda_add_executable(cmtl_disordered_lattice cmtl_disordered_lattice.cu
	OPTIONS "-DMTL_WITH_SHORT_ELE_PROD;-DMTL_HAS_ODEINT;-DMTL_HAS_CUDA")
endif (CUDA_FOUND)

add_executable(vexcl_disordered_lattice vexcl_disordered_lattice.cpp)
target_link_libraries(vexcl_disordered_lattice OpenCL ${Boost_LIBRARIES})
set_target_properties(vexcl_disordered_lattice PROPERTIES COMPILE_FLAGS -std=c++0x)

add_executable(viennacl_disordered_lattice viennacl_disordered_lattice.cpp)
target_link_libraries(viennacl_disordered_lattice OpenCL pugixml ${Boost_LIBRARIES})
set_target_properties(viennacl_disordered_lattice PROPERTIES COMPILE_FLAGS -std=c++0x)

foreach(script run_thrust run_vexcl run_viennacl)
    configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/${script}.sge
	${CMAKE_CURRENT_BINARY_DIR}/${script}.sge
	COPYONLY
	)
endforeach(script)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/run_on_tahiti.sh
    ${CMAKE_CURRENT_BINARY_DIR}/run_on_tahiti.sh
    COPYONLY
    )
