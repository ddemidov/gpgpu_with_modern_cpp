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

struct sys_func
{
    const vector_type &R;

    std::vector<cl::Kernel> step;

    sys_func( const vector_type &_R ) : R( _R ) {
	static const char source[] =
	    "#if defined(cl_khr_fp64)\n"
	    "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
	    "#elif defined(cl_amd_fp64)\n"
	    "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
	    "#endif\n"
	    "kernel void lorenz(\n"
	    "        ulong n,\n"
	    "        global double *dx,\n"
	    "        global double *dy,\n"
	    "        global double *dz,\n"
	    "        global const double *x,\n"
	    "        global const double *y,\n"
	    "        global const double *z,\n"
	    "        global const double *r,\n"
	    "        double sigma,\n"
	    "        double b\n"
	    "        )\n"
	    "{\n"
	    "    size_t i = get_global_id(0);\n"
	    "    if (i < n) {\n"
	    "        double X = x[i];\n"
	    "        double Y = y[i];\n"
	    "        double Z = z[i];\n"
	    "        double R = r[i];\n"
	    "        dx[i] = sigma * (Y - X);\n"
	    "        dy[i] = R * X - Y - X * Z;\n"
	    "        dz[i] = X * Y - b * Z;\n"
	    "    }\n"
	    "}\n";
	    
	step.reserve(R.queue_list().size());
	for(auto &q : R.queue_list()) {
	    cl::Context context = q.getInfo<CL_QUEUE_CONTEXT>();
	    cl::Program program = vex::build_sources(context, source);
	    step.emplace_back(program, "lorenz");
	}
    }

    void operator()( const state_type &x , state_type &dxdt , value_type t )
    {
	auto &queue = R.queue_list();
	for(uint d = 0; d < queue.size(); d++) {
	    if (size_t psize = R.part_size(d)) {
		step[d].setArg(0, psize);
		step[d].setArg(1, dxdt(0)(d));
		step[d].setArg(2, dxdt(1)(d));
		step[d].setArg(3, dxdt(2)(d));
		step[d].setArg(4, x(0)(d));
		step[d].setArg(5, x(1)(d));
		step[d].setArg(6, x(2)(d));
		step[d].setArg(7, R(d));
		step[d].setArg(8, sigma);
		step[d].setArg(9, b);

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
    n = argc > 1 ? atoi(argv[1]) : 1024;
    using namespace std;

    vex::Context ctx( vex::Filter::Exclusive( vex::Filter::Env ) );
    std::cout << ctx << std::endl;



    value_type Rmin = 0.1 , Rmax = 50.0 , dR = ( Rmax - Rmin ) / value_type( n - 1 );
    std::vector<value_type> r( n );
    for( size_t i=0 ; i<n ; ++i ) r[i] = Rmin + dR * value_type( i );

    state_type X(ctx.queue(), n);
    X = 10.0;

    vector_type R( ctx.queue() , r );

    odeint::runge_kutta4<
	    state_type , value_type , state_type , value_type ,
	    odeint::vector_space_algebra , odeint::default_operations
	    > stepper;

    odeint::integrate_const( stepper , sys_func( R ) , X , value_type(0.0) , t_max , dt );

    std::vector< value_type > res( n );
    vex::copy( X(0) , res );
    //for( size_t i=0 ; i<n ; ++i )
    //	cout << res[i] << "\t" << r[i] << "\n";
    cout << res[0] << endl;

}
