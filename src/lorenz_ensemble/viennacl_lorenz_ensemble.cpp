#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <memory>

#include <vexcl/vexcl.hpp>
#include <viennacl/vector.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/fusion_algebra.hpp>
#include <boost/fusion/sequence/intrinsic/at_c.hpp>

#include <boost/numeric/odeint/external/viennacl/viennacl_operations.hpp>

namespace odeint = boost::numeric::odeint;
namespace fusion = boost::fusion;

typedef double value_type;

typedef fusion::vector<
    viennacl::vector< value_type > ,
    viennacl::vector< value_type > ,
    viennacl::vector< value_type >
    > state_type;

namespace boost { namespace numeric { namespace odeint {

template<>
struct is_resizeable< state_type > : boost::true_type { };

template<>
struct resize_impl< state_type , state_type >
{
    static void resize( state_type &x1 , const state_type &x2 )
    {
        fusion::at_c< 0 >( x1 ).resize( fusion::at_c< 0 >( x2 ).size() , false );
        fusion::at_c< 1 >( x1 ).resize( fusion::at_c< 1 >( x2 ).size() , false );
        fusion::at_c< 2 >( x1 ).resize( fusion::at_c< 2 >( x2 ).size() , false );
    }
};

template<>
struct same_size_impl< state_type , state_type >
{
    static bool same_size( const state_type &x1 , const state_type &x2 )
    {
        return
            ( fusion::at_c< 0 >( x1 ).size() == fusion::at_c< 0 >( x2 ).size() ) &&
            ( fusion::at_c< 1 >( x1 ).size() == fusion::at_c< 1 >( x2 ).size() ) &&
            ( fusion::at_c< 2 >( x1 ).size() == fusion::at_c< 2 >( x2 ).size() ) ;
    }
};

} } }

const value_type sigma = 10.0;
const value_type b = 8.0 / 3.0;

struct sys_func
{
    const viennacl::vector<value_type> &R;

    sys_func( const viennacl::vector<value_type> &_R ) : R( _R ) {
	static const char source[] =
	    "#if defined(cl_khr_fp64)\n"
	    "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
	    "#elif defined(cl_amd_fp64)\n"
	    "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
	    "#endif\n"
	    "kernel void lorenz(\n"
	    "        uint n,\n"
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
	    "    for(uint i = get_global_id(0); i < n; i += get_global_size(0)) {\n"
	    "        double X = x[i];\n"
	    "        double Y = y[i];\n"
	    "        double Z = z[i];\n"
	    "        double R = r[i];\n"
	    "        dx[i] = sigma * (Y - X);\n"
	    "        dy[i] = R * X - Y - X * Z;\n"
	    "        dz[i] = X * Y - b * Z;\n"
	    "    }\n"
	    "}\n";

	viennacl::ocl::current_context().add_program(
		source, "lorenz_program").add_kernel("lorenz");
    }

    void operator()( const state_type &x , state_type &dxdt , value_type t ) const
    {
        const viennacl::vector< value_type > &X = fusion::at_c< 0 >( x );
        const viennacl::vector< value_type > &Y = fusion::at_c< 1 >( x );
        const viennacl::vector< value_type > &Z = fusion::at_c< 2 >( x );

	viennacl::vector< value_type > &dX = fusion::at_c<0>( dxdt );
	viennacl::vector< value_type > &dY = fusion::at_c<1>( dxdt );
	viennacl::vector< value_type > &dZ = fusion::at_c<2>( dxdt );

	viennacl::ocl::kernel &step =
	    viennacl::ocl::current_context().get_program(
		    "lorenz_program").get_kernel("lorenz");

	viennacl::ocl::enqueue(
		step(cl_uint(X.size()),
		     dX, dY, dZ,
		     X, Y, Z,
		     R, sigma, b
		     )
		);
    }
};

size_t n;
const value_type dt = 0.01;
const value_type t_max = 100.0;

int main( int argc , char **argv )
{
    using namespace std;

    n = argc > 1 ? atoi( argv[1] ) : 1024;

    vex::Context ctx( vex::Filter::Env && vex::Filter::Count(1));

    std::vector<cl_device_id> dev_id(1, ctx.queue(0).getInfo<CL_QUEUE_DEVICE>()());
    std::vector<cl_command_queue> queue_id(1, ctx.queue(0)());

    viennacl::ocl::setup_context(0, ctx.context(0)(), dev_id, queue_id);

    cout << ctx << endl;


    value_type Rmin = 0.1 , Rmax = 50.0 , dR = ( Rmax - Rmin ) / value_type( n - 1 );
    std::vector<value_type> r( n );
    std::vector<value_type> tmp( n, 10);
    for( size_t i=0 ; i<n ; ++i ) r[i] = Rmin + dR * value_type( i );

    state_type S;

    viennacl::vector<value_type> &X = fusion::at_c<0>(S);
    viennacl::vector<value_type> &Y = fusion::at_c<1>(S);
    viennacl::vector<value_type> &Z = fusion::at_c<2>(S);

    X.resize( n );
    Y.resize( n );
    Z.resize( n );

    viennacl::copy(tmp, X);
    viennacl::copy(tmp, Y);
    viennacl::copy(tmp, Z);

    viennacl::vector<value_type> R( n );
    viennacl::copy( r , R );

    odeint::runge_kutta4<
	    state_type , value_type , state_type , value_type ,
	    odeint::fusion_algebra , odeint::viennacl_operations
	    > stepper;

    odeint::integrate_const( stepper , sys_func( R ) , S , value_type(0.0) , t_max , dt );

    std::vector< value_type > res( n );
    viennacl::copy( X , res );
    /*
    for( size_t i=0 ; i<n ; ++i )
	cout << res[i] << "\t" << r[i] << "\n";
    */
    cout << res[0] << endl;

    exit(0);
}
