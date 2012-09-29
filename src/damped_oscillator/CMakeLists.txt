cmake_minimum_required(VERSION 2.8)
project(dumped_oscillator)

if (CUDA_FOUND)
    cuda_add_executable(thrust_oscillator thrust_oscillator.cu)

    cuda_add_executable(thrust_oscillator_openmp thrust_oscillator.cu
	OPTIONS "-Xcompiler;-fopenmp;-DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP")
    target_link_libraries(thrust_oscillator_openmp gomp)
endif (CUDA_FOUND)

add_executable(vexcl_oscillator vexcl_oscillator.cpp)
target_link_libraries(vexcl_oscillator OpenCL)
set_target_properties(vexcl_oscillator PROPERTIES COMPILE_FLAGS -std=c++0x)

add_executable(viennacl_oscillator viennacl_oscillator.cpp)
target_link_libraries(viennacl_oscillator OpenCL)
set_target_properties(viennacl_oscillator PROPERTIES COMPILE_FLAGS -std=c++0x)

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
