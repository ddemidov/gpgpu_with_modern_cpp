cmake_minimum_required(VERSION 2.8)
project(disordered_ham_lattice)

add_executable(vexcl_disordered_lattice vexcl_disordered_lattice.cpp)
target_link_libraries(vexcl_disordered_lattice OpenCL)
set_target_properties(vexcl_disordered_lattice PROPERTIES COMPILE_FLAGS -std=c++0x)

add_executable(viennacl_disordered_lattice viennacl_disordered_lattice.cpp)
target_link_libraries(viennacl_disordered_lattice OpenCL pugixml)
set_target_properties(viennacl_disordered_lattice PROPERTIES COMPILE_FLAGS -std=c++0x)

foreach(script run_on_cpu run_on_1gpu run_on_2gpu run_on_3gpu)
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
