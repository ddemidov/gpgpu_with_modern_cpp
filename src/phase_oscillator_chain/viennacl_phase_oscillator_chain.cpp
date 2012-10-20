#include <iostream>
#include <vector>
#include <utility>
#include <tuple>

#include <vexcl/vexcl.hpp>
#include <vexcl/exclusive.hpp>
#include <viennacl/vector.hpp>

#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/vexcl/vexcl_resize.hpp>

#include <boost/numeric/odeint/external/viennacl/viennacl_operations.hpp>
#include <boost/numeric/odeint/external/viennacl/viennacl_resize.hpp>


namespace odeint = boost::numeric::odeint;

typedef double value_type;

typedef viennacl::vector< value_type > state_type;

struct sys_func
{
    const state_type &omega;

    sys_func( const state_type &_omega ) : omega( _omega ) {
	static const char source[] =
	    "#if defined(cl_khr_fp64)\n"
	    "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
	    "#elif defined(cl_amd_fp64)\n"
	    "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
	    "#endif\n"
	    "kernel void oscillator(\n"
	    "        uint n,\n"
	    "        global double *dx,\n"
	    "        global const double *x,\n"
	    "        global const double *omega\n"
	    "        )\n"
	    "{\n"
	    "    for(uint i = get_global_id(0); i < n; i += get_global_size(0)) {\n"
	    "        double X0 = x[i];\n"
	    "        double Xl = i > 0 ? x[i - 1] : X0;\n"
	    "        double Xr = i + 1 < n ? x[i + 1] : X0;\n"
	    "        dx[i] = omega[i] + sin(Xl - X0) + sin(X0 - Xr);\n"
	    "    }\n"
	    "}\n";

	viennacl::ocl::current_context().add_program(
		source, "oscillator_program").add_kernel("oscillator");
    }

    void operator()( const state_type &x , state_type &dxdt , value_type t ) const {
	viennacl::ocl::kernel &step = viennacl::ocl::get_kernel("oscillator_program", "oscillator");

	viennacl::ocl::enqueue( step(cl_uint(x.size()), dxdt, x, omega) );
    }
};

size_t n;
const value_type dt = 0.01;
const value_type t_max = 100.0;
const value_type pi = 3.1415926535897932384626433832795029;

int main( int argc , char **argv )
{
    n = ( argc > 1 ) ? atoi(argv[1]) : 1024;
    const value_type epsilon = 6.0 / ( n * n ); // should be < 8/N^2 to see phase locking
    using namespace std;

    vex::Context ctx( vex::Filter::Exclusive( vex::Filter::Env && vex::Filter::Count(1)) );
    std::vector<cl_device_id> dev_id(1, ctx.queue(0).getInfo<CL_QUEUE_DEVICE>()());
    std::vector<cl_command_queue> queue_id(1, ctx.queue(0)());
    viennacl::ocl::setup_context(0, ctx.context(0)(), dev_id, queue_id);
    cout << ctx << endl;

    // initialize omega and the state of the lattice

    std::vector< value_type > omega( n );
    std::vector< value_type > x( n );
    for( size_t i=0 ; i<n ; ++i )
    {
        x[i] = 2.0 * pi * drand48();
        omega[i] = double( n - i ) * epsilon; // decreasing frequencies
    }

    state_type X( n );
    state_type Omega( n );

    viennacl::copy(x, X);
    viennacl::copy(omega, Omega);

    odeint::runge_kutta4<
	    state_type , value_type , state_type , value_type ,
	    odeint::vector_space_algebra , odeint::viennacl_operations
	    > stepper;

    sys_func sys(Omega);
    odeint::integrate_const( stepper , std::ref( sys ) , X , value_type( 0.0 ) , t_max , dt );

    std::vector< value_type > res( n );
    viennacl::copy( X , res );
    cout << res[0] << endl;
//    for( size_t i=0 ; i<n ; ++i ) cout << res[i] << endl;

    exit(0);
}
