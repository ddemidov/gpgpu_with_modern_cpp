#include <iostream>
#include <vector>
#include <array>
#include <utility>
#include <tuple>
#include <memory>

//#define VEXCL_SHOW_KERNELS
#include <vexcl/vexcl.hpp>
#include <vexcl/exclusive.hpp>
#include <vexcl/generator.hpp>

#include <boost/numeric/odeint.hpp>

namespace odeint = boost::numeric::odeint;
namespace fusion = boost::fusion;

typedef double value_type;

typedef vex::generator::symbolic< value_type > sym_vector;

typedef std::array<sym_vector, 3> sym_state;

const value_type sigma = 10.0;
const value_type b = 8.0 / 3.0;

struct sys_func
{
    const sym_vector &R;

    sys_func( const sym_vector &_R ) : R( _R ) {}

    void operator()( const sym_state &x , sym_state &dxdt , value_type t ) const
    {
	dxdt[0] = sigma * (x[1] - x[0]);
	dxdt[1] = R * x[0] - x[1] - x[0] * x[2];
	dxdt[2] = x[0] * x[1] - b * x[2];
    }
};

size_t n;
const value_type dt = 0.01;
const value_type t_max = 100.0;

int main( int argc , char **argv )
{
    using namespace std;

    n = argc > 1 ? atoi( argv[1] ) : 1024;

    vex::Context ctx( vex::Filter::Exclusive( vex::Filter::DoublePrecision && vex::Filter::Env ) );
    cout << ctx << endl;


    // Custom kernel body will be recorded here:
    std::ostringstream body;
    vex::generator::set_recorder(body);

    // State types that would become kernel parameters:
    sym_state  sym_S = {{
	sym_vector::Parameter,
	sym_vector::Parameter,
	sym_vector::Parameter
    }};

    sym_vector sym_R(sym_vector::Parameter, sym_vector::Vector, sym_vector::Const);

    // Symbolic stepper:
    odeint::runge_kutta4<
	    sym_state , value_type , sym_state , value_type ,
	    odeint::range_algebra , odeint::default_operations
	    > sym_stepper;

    sys_func sys(sym_R);
    sym_stepper.do_step(std::ref(sys), sym_S, 0, dt);

    auto kernel = vex::generator::build_kernel(ctx.queue(), "lorenz", body.str(),
	    sym_S[0], sym_S[1], sym_S[2], sym_R
	    );

    // Real state initialization:
    value_type Rmin = 0.1 , Rmax = 50.0 , dR = ( Rmax - Rmin ) / value_type( n - 1 );
    std::vector<value_type> r( n );
    for( size_t i=0 ; i<n ; ++i ) r[i] = Rmin + dR * value_type( i );

    vex::vector<value_type> X(ctx.queue(), n);
    vex::vector<value_type> Y(ctx.queue(), n);
    vex::vector<value_type> Z(ctx.queue(), n);
    vex::vector<value_type> R(ctx.queue(), r);

    X = 10.0;
    Y = 10.0;
    Z = 10.0;

    // Integration loop:
    for(value_type t = 0; t < t_max; t += dt)
	kernel(X, Y, Z, R);

    std::vector< value_type > res( n );
    vex::copy( X , res );
    cout << res[0] << endl;
}
