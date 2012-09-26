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

    std::vector<cl::Kernel> step;

    oscillator(const std::vector<cl::CommandQueue> &queue,
	    value_type omega, value_type amp, value_type offset, value_type omega_d
	    )
        : m_omega( omega ) , m_amp( amp ) , m_offset( offset ) , m_omega_d( omega_d )
    {
	static const char source[] =
	    "#if defined(cl_khr_fp64)\n"
	    "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
	    "#elif defined(cl_amd_fp64)\n"
	    "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
	    "#endif\n"
	    "kernel void oscillator(\n"
	    "        ulong n,\n"
	    "        global double *dx,\n"
	    "        global double *dy,\n"
	    "        global const double *x,\n"
	    "        global const double *y,\n"
	    "        double omega,\n"
	    "        double eps\n"
	    "        )\n"
	    "{\n"
	    "    size_t i = get_global_id(0);\n"
	    "    if (i < n) {\n"
	    "        double X = x[i];\n"
	    "        double Y = y[i];\n"
	    "        dx[i] = eps * X + omega * Y;\n"
	    "        dy[i] = eps * Y - omega * X;\n"
	    "    }\n"
	    "}\n";
	    
	step.reserve(queue.size());
	for(auto &q : queue) {
	    cl::Context context = q.getInfo<CL_QUEUE_CONTEXT>();
	    cl::Program program = vex::build_sources(context, source);
	    step.emplace_back(program, "oscillator");
	}
    }

    void operator()( const state_type &x , state_type &dxdt , value_type t )
    {
	auto &queue = x(0).queue_list();

        value_type eps = m_offset + m_amp * cos( m_omega_d * t );

	for(uint d = 0; d < queue.size(); d++) {
	    if (size_t psize = x(0).part_size(d)) {
		step[d].setArg(0, psize);
		step[d].setArg(1, dxdt(0)(d));
		step[d].setArg(2, dxdt(1)(d));
		step[d].setArg(3, x(0)(d));
		step[d].setArg(4, x(1)(d));
		step[d].setArg(5, m_omega);
		step[d].setArg(6, eps);

		queue[d].enqueueNDRangeKernel(step[d], cl::NullRange, psize, cl::NullRange);
	    }
	}
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
