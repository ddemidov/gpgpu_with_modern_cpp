/*
 * phase_osc_chain.cu
 *
 *  Created on: Apr 1, 2011
 *      Author: mario
 */

/*
 * This example shows how to use odeint on CUDA devices with thrust.
 * Note that we require at least Version 3.2 of the nVidia CUDA SDK
 * and the thrust library should be installed in the CUDA include
 * folder.
 *
 * As example we use a chain of phase oscillators with nearest neighbour
 * coupling, as described in:
 *
 * Avis H. Cohen, Philip J. Holmes and Richard H. Rand:
 * JOURNAL OF MATHEMATICAL BIOLOGY Volume 13, Number 3, 345-369,
 *
 */

#include <iostream>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/external/thrust/thrust_algebra.hpp>
#include <boost/numeric/odeint/external/thrust/thrust_operations.hpp>
#include <boost/numeric/odeint/external/thrust/thrust_resize.hpp>

using namespace std;

using namespace boost::numeric::odeint;


//change this to float if your device does not support double computation
typedef double value_type;


typedef thrust::device_vector< value_type > state_type;
typedef thrust::device_vector< size_t > index_vector_type;


class phase_oscillators
{

public:

    struct sys_functor
    {
        template< class Tuple >
        __host__ __device__
        void operator()( Tuple t )  // this functor works on tuples of values
        {
            // first, unpack the tuple into value, neighbors and omega
            const value_type phi = thrust::get<0>(t);
            const value_type phi_left = thrust::get<1>(t);  // left neighbor
            const value_type phi_right = thrust::get<2>(t); // right neighbor
            const value_type omega = thrust::get<3>(t);
            // the dynamical equation
            thrust::get<4>(t) = omega + sin( phi_right - phi ) + sin( phi - phi_left );
        }
    };

    phase_oscillators( const state_type &omega )
        : m_omega( omega ) , m_N( omega.size() ) , m_prev( omega.size() ) , m_next( omega.size() )
    {
        // build indices pointing to left and right neighbours
        thrust::counting_iterator<size_t> c( 0 );
        thrust::copy( c , c+m_N-1 , m_prev.begin()+1 );
        m_prev[0] = 0; // m_prev = { 0 , 0 , 1 , 2 , 3 , ... , N-1 }

        thrust::copy( c+1 , c+m_N , m_next.begin() );
        m_next[m_N-1] = m_N-1; // m_next = { 1 , 2 , 3 , ... , N-1 , N-1 }
    }

    void operator() ( const state_type &x , state_type &dxdt , const value_type dt )
    {
        thrust::for_each(
                thrust::make_zip_iterator(
                        thrust::make_tuple(
                                x.begin() ,
                                thrust::make_permutation_iterator( x.begin() , m_prev.begin() ) ,
                                thrust::make_permutation_iterator( x.begin() , m_next.begin() ) ,
                                m_omega.begin() ,
                                dxdt.begin()
                                ) ),
                thrust::make_zip_iterator(
                        thrust::make_tuple(
                                x.end() ,
                                thrust::make_permutation_iterator( x.begin() , m_prev.end() ) ,
                                thrust::make_permutation_iterator( x.begin() , m_next.end() ) ,
                                m_omega.end() ,
                                dxdt.end()) ) ,
                sys_functor()
                );
    }

private:

    const state_type &m_omega;
    const size_t m_N;
    index_vector_type m_prev;
    index_vector_type m_next;
};





size_t n;
const value_type dt = 0.01;
const value_type t_max = 100.0;

int main( int argc , char* argv[] )
{
    n = ( argc > 1 ) ? atoi(argv[1]) : 1024;
    const value_type epsilon = 6.0 / ( n * n ); // should be < 8/N^2 to see phase locking

    vector< value_type > x_host( n );
    vector< value_type > omega_host( n );
    for( size_t i=0 ; i<n ; ++i )
    {
        x_host[i] = 2.0 * M_PI * drand48();
        omega_host[i] = double( n - i ) * epsilon; // decreasing frequencies
    }

    state_type x = x_host;
    state_type omega = omega_host;

    runge_kutta4< state_type , value_type , state_type , value_type , thrust_algebra , thrust_operations > stepper;

    phase_oscillators sys( omega );

    integrate_const( stepper , sys , x , 0.0 , t_max , dt );

    std::vector< value_type > res( n );
    thrust::copy( x.begin() , x.end() , res.begin() );
    cout << res[0] << endl;
}
