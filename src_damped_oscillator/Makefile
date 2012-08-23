NVCC          = /usr/local/cuda/bin/nvcc
VEXCL_ROOT    = /home/karsten/boost/vexcl
CUDA_INC      = /usr/local/cuda/include
VIENNACL_ROOT = /home/karsten/boost/testing/ViennaCL-1.3.0
# ODEINT_ROOT   = $(ODEINT_ROOT)
# BOOST_ROOT    = $(BOOST_ROOT)


all: thrust_oscillator vexcl_oscillator thrust_oscillator_openmp oscillator

oscillator : oscillator.cpp
	g++ -o $@ -std=c++0x -I$(ODEINT_ROOT) -I$(BOOST_ROOT) -O3 $^

thrust_oscillator: thrust_oscillator.cu
	$(NVCC) -o $@ -I$(ODEINT_ROOT) -I$(BOOST_ROOT) -arch=sm_13 -O3 $^

thrust_oscillator_openmp: thrust_oscillator.cu
	$(NVCC) -o $@ -I$(ODEINT_ROOT) -I$(BOOST_ROOT) -O3 -Xcompiler -fopenmp -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP $^

vexcl_oscillator: vexcl_oscillator.cpp
	g++ -o $@ -std=c++0x -I$(ODEINT_ROOT) -I$(BOOST_ROOT) -I$(VEXCL_ROOT) -I$(CUDA_INC) -O3 $^ -lOpenCL

viennacl_oscillator : viennacl_oscillator.cpp
	g++ -o $@ -std=c++0x -I$(ODEINT_ROOT) -I$(BOOST_ROOT) -I$(VIENNACL_ROOT) -I$(CUDA_INC) -O3 $^ -lOpenCL

.PHONY: clean

clean: thrust_oscillator vexcl_oscillator thrust_oscillator_openmp oscillator
	rm -f $^
