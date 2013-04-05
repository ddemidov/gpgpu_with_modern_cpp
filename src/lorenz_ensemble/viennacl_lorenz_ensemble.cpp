#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <memory>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/fusion_algebra.hpp>
#include <boost/fusion/sequence/intrinsic/at_c.hpp>

#include <boost/numeric/odeint/external/viennacl/viennacl_operations.hpp>
#include <boost/numeric/odeint/external/viennacl/viennacl_resize.hpp>

#include <vexcl/vexcl.hpp>
#include <viennacl/vector.hpp>

namespace odeint = boost::numeric::odeint;
namespace fusion = boost::fusion;

typedef double value_type;

typedef fusion::vector<
    viennacl::vector< value_type > ,
    viennacl::vector< value_type > ,
    viennacl::vector< value_type >
    > state_type;

const value_type sigma = 10.0;
const value_type b = 8.0 / 3.0;

struct sys_func
{
    const viennacl::vector<value_type> &R;

    sys_func( const viennacl::vector<value_type> &_R ) : R( _R ) {
    }

    void operator()( const state_type &x , state_type &dxdt , value_type t ) const
    {
        typedef viennacl::generator::vector<value_type> vec;

	const auto &X = fusion::at_c< 0 >( x );
	const auto &Y = fusion::at_c< 1 >( x );
        const auto &Z = fusion::at_c< 2 >( x );

	auto &dX = fusion::at_c<0>( dxdt );
	auto &dY = fusion::at_c<1>( dxdt );
	auto &dZ = fusion::at_c<2>( dxdt );

	viennacl::generator::custom_operation op;

        op.add(vec(dX) = sigma * (vec(Y) - vec(X)));
        op.add(vec(dY) = element_prod(vec(R), vec(X)) - vec(Y) - element_prod(vec(X), vec(Z)));
        op.add(vec(dZ) = element_prod(vec(X), vec(Y)) - b * vec(Z));

        op.execute();
    }
};

size_t n;
const value_type dt = 0.01;
const value_type t_max = 100.0;

int main( int argc , char **argv )
{
    using namespace std;

    n = argc > 1 ? atoi( argv[1] ) : 1024;

    vex::Context ctx( vex::Filter::Exclusive( vex::Filter::Env && vex::Filter::Count(1)) );

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
