cmake_minimum_required(VERSION 2.8)
project(lorenz_ensemble)

cuda_add_executable(thrust_lorenz thrust_lorenz_ensemble.cu)

cuda_add_executable(thrust_lorenz_openmp thrust_lorenz_ensemble.cu
    OPTIONS "-Xcompiler;-fopenmp;-DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP")
target_link_libraries(thrust_lorenz_openmp gomp)

add_executable(vexcl_lorenz vexcl_lorenz_ensemble.cpp)
target_link_libraries(vexcl_lorenz OpenCL)
set_target_properties(vexcl_lorenz PROPERTIES COMPILE_FLAGS -std=c++0x)

add_executable(viennacl_lorenz viennacl_lorenz_ensemble.cpp)
target_link_libraries(viennacl_lorenz OpenCL)
set_target_properties(viennacl_lorenz PROPERTIES COMPILE_FLAGS -std=c++0x)
