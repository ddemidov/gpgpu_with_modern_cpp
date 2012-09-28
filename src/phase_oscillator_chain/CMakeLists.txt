cmake_minimum_required(VERSION 2.8)
project(phase_oscillator_chain)

if (CUDA_FOUND)
    cuda_add_executable(thrust_phase_oscillator thrust_phase_oscillator_chain.cu)

    cuda_add_executable(thrust_phase_oscillator_openmp thrust_phase_oscillator_chain.cu
	OPTIONS "-Xcompiler;-fopenmp;-DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP")
    target_link_libraries(thrust_phase_oscillator_openmp gomp)
endif (CUDA_FOUND)

add_executable(vexcl_phase_oscillator vexcl_phase_oscillator_chain.cpp)
target_link_libraries(vexcl_phase_oscillator OpenCL)
set_target_properties(vexcl_phase_oscillator PROPERTIES COMPILE_FLAGS -std=c++0x)

add_executable(viennacl_phase_oscillator viennacl_phase_oscillator_chain.cpp)
target_link_libraries(viennacl_phase_oscillator OpenCL)
set_target_properties(viennacl_phase_oscillator PROPERTIES COMPILE_FLAGS -std=c++0x)

foreach(script run_on_cpu run_on_1gpu run_on_2gpu run_on_3gpu)
    configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/${script}.sge
	${CMAKE_CURRENT_BINARY_DIR}/${script}.sge
	COPYONLY
	)
endforeach(script)
