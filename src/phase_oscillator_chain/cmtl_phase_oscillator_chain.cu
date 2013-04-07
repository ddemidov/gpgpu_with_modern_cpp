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
#include <boost/numeric/mtl/cuda/cuda_utility.hpp>

#include <boost/numeric/mtl/mtl.hpp>

#include <boost/numeric/mtl/interface/odeint.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/vector_space_algebra.hpp>

#include <boost/timer.hpp>


namespace odeint = boost::numeric::odeint;

using namespace mtl;


struct stencil_kernel
{
    typedef double value_type;
    static const int start= -1, end= 1;

    stencil_kernel(int n) : n(n) {}

    template <typename Vector>
    __device__ __host__ value_type operator()(const Vector& v, int i) const
    {
	return sin(v[i+1] - v[i]) + sin(v[i] - v[i-1]);
    }

    template <typename Vector>
    __device__ __host__ value_type outer_stencil(const Vector& v, int i, int offset= 0) const
    {
	value_type s1= i > offset? sin(v[i] - v[i-1]) : sin(v[i]), 
	           s2= i+1 < n + offset? sin(v[i+1] - v[i]) : sin(v[i]);
	return s1 + s2;
    }

    int n;
};

template <typename State>
struct sys_func
{
    typedef typename Collection<State>::value_type value_type;

    sys_func(const State& omega) 
      : omega(omega), S(num_rows(omega)) {}

    void operator()(const State &x, State &dxdt, value_type t) const
    {
	dxdt = S * x;
        dxdt += omega;
    }

    const State&   omega;
    mtl::matrix::stencil1D<stencil_kernel> S; 
};


int main(int argc, char* argv[])
{
    using namespace mtl;
    mtl::vampir_trace<9999>                            tracer;

    typedef double                    value_type;
    typedef dense_vector<value_type>  state_type;

    const value_type dt= 0.01, pi= M_PI, t_max= 100.0;
    const size_t n= argc > 1 ? atoi(argv[1]) : 1024;
    const value_type epsilon = 6.0 / ( n * n ); // should be < 8/N^2 to see phase locking

    state_type omega(n), x(n), tmp(n);
    for (size_t i= 0; i < n; ++i) {
        x[i] = 2.0 * pi * drand48();
        omega[i] = double(n - i) * epsilon; // decreasing frequencies
    }

    odeint::runge_kutta4<
	    state_type, value_type, state_type, value_type,
	    odeint::vector_space_algebra, odeint::default_operations
	    > stepper;

    sys_func<state_type> sys(omega);
    boost::timer timer;
    odeint::integrate_const(stepper, boost::ref(sys), x, 0.0, t_max, dt);
    cudaThreadSynchronize();
    std::cout << "Integration took " << timer.elapsed() << " s\n";
    
    std::cout << "Result is " << x[0] << '\n';

    return 0;
}

