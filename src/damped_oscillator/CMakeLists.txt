cmake_minimum_required(VERSION 2.8)
project(dumped_oscillator)

cuda_add_executable(thrust_oscillator thrust_oscillator.cu)

cuda_add_executable(thrust_oscillator_openmp thrust_oscillator.cu
    OPTIONS "-Xcompiler;-fopenmp;-DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP")
target_link_libraries(thrust_oscillator_openmp gomp)

add_executable(vexcl_oscillator vexcl_oscillator.cpp)
target_link_libraries(vexcl_oscillator OpenCL)
set_target_properties(vexcl_oscillator PROPERTIES COMPILE_FLAGS -std=c++0x)

add_executable(viennacl_oscillator viennacl_oscillator.cpp)
target_link_libraries(viennacl_oscillator OpenCL)
set_target_properties(viennacl_oscillator PROPERTIES COMPILE_FLAGS -std=c++0x)
