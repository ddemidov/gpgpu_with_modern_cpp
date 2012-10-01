cmake_minimum_required(VERSION 2.8)
project(lorenz_ensemble)

if (CUDA_FOUND)
    cuda_add_executable(thrust_lorenz thrust_lorenz_ensemble.cu)

    cuda_add_executable(thrust_lorenz_openmp thrust_lorenz_ensemble.cu
	OPTIONS "-Xcompiler;-fopenmp;-DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP")
    target_link_libraries(thrust_lorenz_openmp gomp)
endif (CUDA_FOUND)

add_executable(vexcl_lorenz vexcl_lorenz_ensemble.cpp)
target_link_libraries(vexcl_lorenz OpenCL)
set_target_properties(vexcl_lorenz PROPERTIES COMPILE_FLAGS -std=c++0x)

add_executable(viennacl_lorenz viennacl_lorenz_ensemble.cpp)
target_link_libraries(viennacl_lorenz OpenCL)
set_target_properties(viennacl_lorenz PROPERTIES COMPILE_FLAGS -std=c++0x)

add_executable(generated_lorenz generated_lorenz_ensemble.cpp)
target_link_libraries(generated_lorenz OpenCL)
set_target_properties(generated_lorenz PROPERTIES COMPILE_FLAGS -std=c++0x)

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
