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

#include <iostream>
#include <complex>
#include <vector>

#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/utility/enable_if.hpp>

#include <boost/numeric/mtl/mtl.hpp>

#include <boost/numeric/mtl/interface/odeint.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/vector_space_algebra.hpp>

#include <boost/timer.hpp>

namespace odeint = boost::numeric::odeint;

template <typename value_type>
struct sys_func
{
    typedef mtl::dense_vector<value_type>         vector_type;
    typedef mtl::multi_vector<vector_type>        state_type;
    // typedef boost::fusion::vector<vector_type, vector_type, vector_type>      state_type;

    explicit sys_func(const vector_type &R) : R(R) { }

    void operator()(const state_type& x, state_type& dxdt, value_type)
    {
	dxdt.at(0)= sigma * (x.at(1) - x.at(0));
	dxdt.at(1)= R * x.at(0) - x.at(1) - x.at(0) * x.at(2);
	dxdt.at(2)= x.at(0) * x.at(1) - b * x.at(2);
    }

  private:
    const vector_type &R;
    static const value_type sigma, b;
    // static const value_type sigma = 10.0, b = 8.0 / 3.0;
};

template <typename value_type>
const value_type sys_func<value_type>::sigma= 10.0;

template <typename value_type>
const value_type sys_func<value_type>::b = 8.0 / 3.0;

int main(int argc, char* argv[])
{
    typedef double                                     value_type;
    typedef typename sys_func<value_type>::vector_type vector_type;
    typedef typename sys_func<value_type>::state_type  state_type;

    const size_t     n= argc > 1 ? atoi(argv[1]) : 1024*1024;
    const value_type dt= 0.01, t_max= 100.0, 
                     Rmin= 0.1, Rmax= 50.0, dR= (Rmax - Rmin) / value_type(n - 1);

    vector_type      R(n);
    for (size_t i= 0; i < n; ++i)
	R[i]= Rmin + dR * value_type(i);

    state_type X(vector_type(n, 10.0), 3);

    odeint::runge_kutta4<state_type, value_type, state_type, value_type,
			 odeint::vector_space_algebra , odeint::default_operations> stepper;
    boost::timer timer;
    odeint::integrate_const(stepper, sys_func<value_type>(R), X, value_type(0), t_max, dt);
    std::cout << "Integration took " << timer.elapsed() << " s\n";

    std::cout << "Result = " << X.at(0)[0] << std::endl;
 

    return 0;
}

