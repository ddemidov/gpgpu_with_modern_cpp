#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <random>
#include <algorithm>
#include <functional>
#include <memory>

//#define VEXCL_SHOW_KERNELS
#include <vexcl/vexcl.hpp>
#include <vexcl/exclusive.hpp>
#include <vexcl/generator.hpp>

#include <boost/numeric/odeint.hpp>

namespace odeint = boost::numeric::odeint;
namespace fusion = boost::fusion;

typedef double value_type;

typedef vex::generator::symbolic<value_type> sym_value;
typedef std::array<sym_value,2> sym_state;

struct oscillator
{
    value_type m_omega;
    value_type m_amp;
    value_type m_offset;
    value_type m_omega_d;

    sym_value &sym_time;

    oscillator(value_type omega, value_type amp, value_type offset, value_type omega_d,
	    sym_value &sym_time)
        : m_omega(omega), m_amp(amp) , m_offset(offset) , m_omega_d(omega_d),
	  sym_time(sym_time)
    {
    }

    void operator()( const sym_state &x , sym_state &dxdt , value_type t )
    {
	using namespace vex;

        sym_value eps;
	// This function would be run for the first iteration only.
	// Integrattion step there begins at time=0, so value of parameter t
	// may be considered as relative to current time. We use this fact to
	// link this time shift with symbolic time that would later come as
	// kernel parameter.
	eps = m_offset + m_amp * cos( m_omega_d * (sym_time + t) );

	dxdt[0] = eps * x[0] + m_omega * x[1];
	dxdt[1] = eps * x[1] - m_omega * x[0];
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
	sym_value::VectorParameter,
	sym_value::VectorParameter
    }};
    sym_value sym_time(sym_value::ScalarParameter);

    // Symbolic stepper:
    odeint::runge_kutta4<
	    sym_state , value_type , sym_state , value_type ,
	    odeint::range_algebra , odeint::default_operations
	    > sym_stepper;

    oscillator sys(1.0, 0.2, 0.0, 1.2, sym_time);
    sym_stepper.do_step(std::ref(sys), sym_S, 0, dt);

    auto kernel = vex::generator::build_kernel(ctx.queue(),
	    "damped_oscillator", body.str(),
	    sym_S[0], sym_S[1], sym_time
	    );

    // Actual data.
    std::vector<value_type> x( 2 * n );
    std::generate( x.begin() , x.end() , drand48 );

    vex::vector<value_type> X(ctx.queue(), n);
    vex::vector<value_type> Y(ctx.queue(), n);

    vex::copy( x.begin() , x.begin() + n, X.begin() );
    vex::copy( x.begin() + n, x.end() , Y.begin() );

    // Integration loop:
    for(value_type t = 0; t < t_max; t += dt)
	kernel(X, Y, t);

    std::vector< value_type > res( n );
    vex::copy( X , res );
    cout << res[0] << endl;
}
