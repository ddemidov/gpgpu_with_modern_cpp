#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <random>
#include <algorithm>
#include <functional>

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

    oscillator( value_type omega = 1.0 , value_type amp = 0.5 , value_type offset = 0.0 , value_type omega_d = 1.2 )
        : m_omega( omega ) , m_amp( amp ) , m_offset( offset ) , m_omega_d( omega_d ) { }

    void operator()( const state_type &x , state_type &dxdt , value_type t ) const
    {
        value_type eps = m_offset + m_amp * cos( m_omega_d * t );
        dxdt(0) =  m_omega * x(1) + eps * x(0);
        dxdt(1) = -m_omega * x(0) + eps * x(1);
    }
};


size_t n;
const value_type dt = 0.01;
const value_type t_max = 100.0;

int main( int argc , char **argv )
{
    n = argc > 1 ? atoi(argv[1]) : 1024;
    using namespace std;

    vex::Context ctx( vex::Filter::DoublePrecision && vex::Filter::Env );
    std::cout << ctx << std::endl;


    std::mt19937 rng;
    std::normal_distribution< value_type > gauss( 0.0 , 1.0 );
    std::vector<value_type> x( n ) , y( n );
    std::generate( x.begin() , x.end() , std::bind( gauss , std::ref( rng ) ) );
    std::generate( y.begin() , y.end() , std::bind( gauss , std::ref( rng ) ) );

    state_type X(ctx.queue(), n);
    vex::copy( x , X(0) );
    vex::copy( y , X(1) );


    odeint::runge_kutta4<
	    state_type , value_type , state_type , value_type ,
	    odeint::vector_space_algebra , odeint::default_operations
	    > stepper;

    odeint::integrate_const( stepper , oscillator( 1.0 , 0.2 , 0.0 , 1.2 ) , X , value_type(0.0) , t_max , dt );

    std::vector< value_type > res( 2 * n );
    vex::copy( X(0) , res );

    //for( size_t i=0 ; i<n ; ++i )
    //	cout << res[i] << "\t" << r[i] << "\n";
    cout << res[0] << endl;

}
