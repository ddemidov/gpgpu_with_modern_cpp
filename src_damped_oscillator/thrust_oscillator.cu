#include <iostream>
#include <cmath>
#include <utility>
#include <cstdlib>


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


const value_type sigma = 10.0;
const value_type b = 8.0 / 3.0;


struct lorenz_system
{
    struct lorenz_functor
    {
        template< class T >
        __host__ __device__
        void operator()( T t ) const
        {
            value_type R = thrust::get< 3 >( t );
            value_type x = thrust::get< 0 >( t );
            value_type y = thrust::get< 1 >( t );
            value_type z = thrust::get< 2 >( t );
            thrust::get< 4 >( t ) = sigma * ( y - x );
            thrust::get< 5 >( t ) = R * x - y - x * z;
            thrust::get< 6 >( t ) = -b * z + x * y ;

        }
    };

    lorenz_system( size_t N , const state_type &beta )
    : m_N( N ) , m_beta( beta ) { }

    template< class State , class Deriv >
    void operator()(  const State &x , Deriv &dxdt , value_type t ) const
    {
        thrust::for_each(
                thrust::make_zip_iterator( thrust::make_tuple(
                        boost::begin( x ) ,
                        boost::begin( x ) + m_N ,
                        boost::begin( x ) + 2 * m_N ,
                        m_beta.begin() ,
                        boost::begin( dxdt ) ,
                        boost::begin( dxdt ) + m_N ,
                        boost::begin( dxdt ) + 2 * m_N  ) ) ,
                thrust::make_zip_iterator( thrust::make_tuple(
                        boost::begin( x ) + m_N ,
                        boost::begin( x ) + 2 * m_N ,
                        boost::begin( x ) + 3 * m_N ,
                        m_beta.begin() ,
                        boost::begin( dxdt ) + m_N ,
                        boost::begin( dxdt ) + 2 * m_N ,
                        boost::begin( dxdt ) + 3 * m_N  ) ) ,
                lorenz_functor() );
    }

    size_t m_N;
    const state_type &m_beta;
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

    vector< value_type > beta_host( N );
    const value_type beta_min = value_type(0.1) , beta_max = value_type(50.0);
    for( size_t i=0 ; i<N ; ++i )
        beta_host[i] = beta_min + value_type( i ) * ( beta_max - beta_min ) / value_type( N - 1 );

    state_type beta = beta_host;


    //[ thrust_lorenz_parameters_integration
    state_type x( 3 * N );

    // initialize x,y,z
    thrust::fill( x.begin() , x.end() , value_type(10.0) );



    typedef runge_kutta4< state_type , value_type , state_type , value_type ,
			  thrust_algebra , thrust_operations > stepper_type;


    lorenz_system lorenz( N , beta );
    integrate_const( stepper_type() , lorenz , x , value_type(0.0) , t_max , dt );

    thrust::host_vector< value_type > res = x;
    // for( size_t i=0 ; i<N ; ++i ) cout << res[i] << "\t" << beta_host[i] << "\n";
    cout << res[0] << endl;



    return 0;
}
