#include <iostream>
#include <vector>
#include <utility>
#include <tuple>

#include <vexcl/vexcl.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/vexcl/vexcl.hpp>

namespace odeint = boost::numeric::odeint;

typedef double value_type;

typedef vex::vector< value_type >    vector_type;
typedef vex::multivector< value_type, 3 > state_type;

const value_type sigma = 10.0;
const value_type b = 8.0 / 3.0;

struct sys_func
{
    const vector_type &R;

    sys_func( const vector_type &_R ) : R( _R ) { }

    void operator()( const state_type &x , state_type &dxdt , value_type t )
    {
	dxdt = std::tie(
		sigma * (x(1) - x(0)),
		R * x(0) - x(1) - x(0) * x(2),
		x(0) * x(1) - b * x(2)
		);
    }
};

size_t n;
const value_type dt = 0.01;
const value_type t_max = 100.0;

int main( int argc , char **argv )
{
    n = 10;
    using namespace std;

    vex::Context ctx( vex::Filter::Exclusive( vex::Filter::Env ) );
    std::cout << ctx << std::endl;



    state_type X(ctx.queue(), n);
    X = 10.0;

    vector_type R( ctx.queue() , n);
    R = (vex::element_index() + 1) * 5;

    odeint::runge_kutta4<
	    state_type , value_type , state_type , value_type ,
	    odeint::vector_space_algebra , odeint::default_operations
	    > stepper;

    odeint::integrate_const( stepper , sys_func( R ) , X , value_type(0.0) , t_max , dt );

    std::cout << "R = " << R << std::endl;
    std::cout << "X = " << X << std::endl;

}
