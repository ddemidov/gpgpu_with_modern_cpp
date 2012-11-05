#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <random>
#include <algorithm>

#include <vexcl/vexcl.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/vector_space_algebra.hpp>
#include <boost/numeric/odeint/external/vexcl/vexcl_resize.hpp>

namespace odeint = boost::numeric::odeint;

typedef double value_type;
typedef vex::vector< value_type >    vector_type;
typedef vex::multivector< value_type, 2 > state_type;

struct oscillator
{
    value_type m_omega;
    value_type m_amp;
    value_type m_offset;
    value_type m_omega_d;

    oscillator(const std::vector<cl::CommandQueue> &queue,
	    value_type omega, value_type amp, value_type offset, value_type omega_d
	    )
        : m_omega( omega ) , m_amp( amp ) , m_offset( offset ) , m_omega_d( omega_d )
    {
    }

    void operator()( const state_type &x , state_type &dxdt , value_type t )
    {
        value_type eps = m_offset + m_amp * cos( m_omega_d * t );

	dxdt = std::tie(
		eps * x(0) + m_omega * x(1),
		eps * x(1) - m_omega * x(0)
		);
    }
};


size_t n;
const value_type dt = 0.01;
const value_type t_max = 100.0;

int main( int argc , char **argv )
{
    try {
    n = argc > 1 ? atoi(argv[1]) : 1024;
    using namespace std;

    vex::Context ctx( vex::Filter::Exclusive( vex::Filter::DoublePrecision && vex::Filter::Env ) );
    std::cout << ctx << std::endl;


    std::vector<value_type> x( 2 * n );
    std::generate( x.begin() , x.end() , drand48 );

    state_type X(ctx.queue(), n);
    vex::copy( x.begin() , x.begin() + n, X(0).begin() );
    vex::copy( x.begin() + n, x.end() , X(1).begin() );


    odeint::runge_kutta4<
	    state_type , value_type , state_type , value_type ,
	    odeint::vector_space_algebra , odeint::default_operations
	    > stepper;

    odeint::integrate_const( stepper , oscillator( ctx.queue(), 1.0 , 0.2 , 0.0 , 1.2 )
	    , X , value_type(0.0) , t_max , dt );

    std::vector< value_type > res( 2 * n );
    vex::copy( X(0) , res );

    //for( size_t i=0 ; i<n ; ++i )
    //	cout << res[i] << "\t" << r[i] << "\n";
    cout << res[0] << endl;

    } catch(const cl::Error &e) {
	using namespace vex;
	std::cout << e << std::endl;
    }
}
