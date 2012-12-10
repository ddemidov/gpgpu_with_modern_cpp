// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

// nvcc can't handle C++11 -> use good ole rand()

#include <cassert>
#include <iostream>
#include <cstdlib>
#include <string>
#include <cmath>
#include <utility>
#include <boost/numeric/mtl/mtl.hpp>

#include <boost/numeric/mtl/interface/odeint.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/vector_space_algebra.hpp>

#include <boost/timer.hpp>


namespace odeint = boost::numeric::odeint;

using namespace mtl;

struct stencil_kernel
{
    static const int start= -1, end= 1;

    template <typename Vector>
    __device__ double inner_stencil(const Vector& v, int i) const
    {
	return v[i-1] + v[i] + v[i+1];
    }

    template <typename Vector>
    __device__ double outer_stencil(const Vector& v, int i, int n) const
    {
	double s= v[i];
	if (i > 0) s+= v[i-1];
	if (i+1 < n) s+= v[i+1];
	return s;
    }
};

template <typename Stencil>
struct test_kernel
{
    test_kernel(const dense_vector<double>& v, dense_vector<double>& w, Stencil stencil) 
      : v(v), wp(w.device_data), n(mtl::vector::size(v)), stencil(stencil) {}

    __device__ void operator()()
    {
	const int size= (0x4000 - 0x28) / sizeof(double);

	__shared__ double tmp[size];
	const unsigned tid= threadIdx.x, bs= blockDim.x;

	for (int i= tid; i < size; i+= bs)
	    tmp[i]= v.dat(i);
	__syncthreads();

	// for (int i= tid; i < n; i+= bs)
	//     stencil.inner_stencil(tmp, i);


#if 1
	wp[tid]= stencil.outer_stencil(tmp, tid, n);

	for (int i= tid + bs; i < n - bs; i+= bs)
	    wp[i]= stencil.inner_stencil(tmp, i);

	wp[n - bs + tid]= stencil.outer_stencil(tmp, n - bs + tid, n);
#endif
    }

    vector::device_expr<dense_vector<double> > v;
    double*                                    wp;
    int                                        n;
    Stencil                                    stencil;
};



int main(int argc, char* argv[])
{
    using namespace mtl;
    mtl::vampir_trace<9999>                            tracer;

    dense_vector<double> v(128), w(128);
    iota(v);

    v.to_device();
    w.to_device();
    test_kernel<stencil_kernel> k(v, w, stencil_kernel());
    launch_function<<<1, 32>>>(k);
    w.to_host();

    std::cout << "w is " << w /*[irange(10)]*/ << '\n';

    return 0;
}

