#include <iostream>
#include <vector>
#include <utility>
#include <tuple>

#include <vexcl/vexcl.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/vector_space_algebra.hpp>
#include <boost/numeric/odeint/external/vexcl/vexcl_resize.hpp>

namespace odeint = boost::numeric::odeint;

typedef double value_type;

typedef vex::vector< value_type >    vector_type;
typedef vex::multivector< value_type, 3 > state_type;

const value_type sigma = 10.0;
const value_type b = 8.0 / 3.0;
const value_type R = 28.0;

struct sys_func
{
    const vector_type &R;

    sys_func( const vector_type &_R ) : R( _R ) { }

    void operator()( const state_type &x , state_type &dxdt , value_type t ) const
    {
	dxdt(0) = -sigma * ( x(0) - x(1) );
	dxdt(1) = R * x(0) - x(1) - x(0) * x(2);
	dxdt(2) = - b * x(2) + x(0) * x(1);
    }
};

size_t n;
const value_type dt = 0.01;
const value_type t_max = 100.0;

int main( int argc , char **argv )
{
    n = argc > 1 ? atoi(argv[1]) : 1024;
    using namespace std;

    vex::Context ctx( vex::Filter::Env );
    std::cout << ctx << std::endl;



    value_type Rmin = 0.1 , Rmax = 50.0 , dR = ( Rmax - Rmin ) / value_type( n - 1 );
    std::vector<value_type> r( n );
    for( size_t i=0 ; i<n ; ++i ) r[i] = Rmin + dR * value_type( i );

    state_type X(ctx.queue(), n);
    X(0) = 10.0;
    X(1) = 10.0;
    X(2) = 10.0;

    vector_type R( ctx.queue() , r );

    odeint::runge_kutta4<
	    state_type , value_type , state_type , value_type ,
	    odeint::vector_space_algebra , odeint::default_operations
	    > stepper;

    odeint::integrate_const( stepper , sys_func( R ) , X , value_type(0.0) , t_max , dt );

    std::vector< value_type > res( 3 * n );
    vex::copy( X(0) , res );
    //for( size_t i=0 ; i<n ; ++i )
    //	cout << res[i] << "\t" << r[i] << "\n";
    // cout << res[0] << endl;

}
