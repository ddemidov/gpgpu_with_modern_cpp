#include <iostream>
#include <cmath>
#include <utility>
#include <cstdlib>
#include <algorithm>


#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include <boost/numeric/odeint.hpp>

#include <boost/numeric/odeint/external/thrust/thrust_algebra.hpp>
#include <boost/numeric/odeint/external/thrust/thrust_operations.hpp>
#include <boost/numeric/odeint/external/thrust/thrust_resize.hpp>



using namespace std;
using namespace boost::numeric::odeint;


typedef double value_type;

//change this to host_vector< ... > of you want to run on CPU
typedef thrust::device_vector< value_type > state_type;
typedef thrust::device_vector< size_t > index_vector_type;
// typedef thrust::host_vector< value_type > state_type;
// typedef thrust::host_vector< size_t > index_vector_type;


struct oscillator
{
    value_type m_omega;
    value_type m_amp;
    value_type m_offset;
    value_type m_omega_d;
    size_t m_N;

    struct oscillator_functor
    {
        value_type m_eps;
        value_type m_omega;
        oscillator_functor( value_type omega , value_type eps )
            : m_omega( omega ) , m_eps( eps ) { }

        template< class T >
        __host__ __device__
        void operator()( T t ) const
        {
            value_type x = thrust::get< 0 >( t );
            value_type y = thrust::get< 1 >( t );
            thrust::get< 2 >( t ) =  m_omega * y + m_eps * x;
            thrust::get< 3 >( t ) = -m_omega * x + m_eps * y;
        }
    };


    oscillator( size_t N , double omega = 1.0 , double amp = 0.5 , double offset = 0.0 , double omega_d = 1.2 )
        : m_N( N ) , m_omega( omega ) , m_amp( amp ) , m_offset( offset ) , m_omega_d( omega_d ) { }

    void operator()( const state_type &x , state_type &dxdt , double t ) const
    {
        double eps = m_offset + m_amp * cos( m_omega_d * t );
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    boost::begin( x ) ,
                    boost::begin( x ) + m_N ,
                    boost::begin( dxdt ) ,
                    boost::begin( dxdt ) + m_N 
                    ) ) ,
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    boost::begin( x ) + m_N ,
                    boost::begin( x ) + 2 * m_N ,
                    boost::begin( dxdt ) + m_N ,
                    boost::begin( dxdt ) + 2 * m_N
                    ) ) ,
            oscillator_functor( m_omega , eps ) );
    }
};



size_t N;
const value_type dt = 0.01;
const value_type t_max = 100.0;


int main( int argc , char* argv[] )
{
    // int driver_version , runtime_version;
    // cudaDriverGetVersion( &driver_version );
    // cudaRuntimeGetVersion ( &runtime_version );
    // cout << driver_version << "\t" << runtime_version << endl;

    N = argc > 1 ? atoi(argv[1]) : 1024;


    std::vector<value_type> x( 2 * N );
    std::generate( x.begin() , x.end() , drand48 );

    state_type X( 2 * N );
    thrust::copy( x.begin() , x.end() , X.begin() );


    typedef runge_kutta4< state_type , value_type , state_type , value_type ,
			  thrust_algebra , thrust_operations > stepper_type;
    integrate_const( stepper_type() , oscillator( 1.0 , 0.2 , 0.0 , 1.2 ) , X , value_type(0.0) , t_max , dt );

    thrust::host_vector< value_type > res = X;
    // for( size_t i=0 ; i<N ; ++i ) cout << res[i] << "\t" << beta_host[i] << "\n";
    cout << res[0] << endl;


    return 0;
}
