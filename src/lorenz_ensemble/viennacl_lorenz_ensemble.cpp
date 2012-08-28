#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <memory>
#include <CL/cl.hpp>

#include <viennacl/vector.hpp>
#include <viennacl/generator/custom_operation.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/fusion_algebra.hpp>
#include <boost/fusion/sequence/intrinsic/at_c.hpp>


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
const value_type R = 28.0;

struct sys_func
{
    const viennacl::vector<value_type> &R;

    std::shared_ptr<viennacl::generator::symbolic_vector    <0, value_type>> dx_d;
    std::shared_ptr<viennacl::generator::symbolic_vector    <1, value_type>> dx_x;
    std::shared_ptr<viennacl::generator::symbolic_vector    <2, value_type>> dx_y;
    std::shared_ptr<viennacl::generator::cpu_symbolic_scalar<3, value_type>> dx_s;
    std::shared_ptr<viennacl::generator::custom_operation> dx_op;

    std::shared_ptr<viennacl::generator::symbolic_vector<0, value_type>> dy_d;
    std::shared_ptr<viennacl::generator::symbolic_vector<1, value_type>> dy_x;
    std::shared_ptr<viennacl::generator::symbolic_vector<2, value_type>> dy_y;
    std::shared_ptr<viennacl::generator::symbolic_vector<3, value_type>> dy_z;
    std::shared_ptr<viennacl::generator::symbolic_vector<4, value_type>> dy_r;
    std::shared_ptr<viennacl::generator::custom_operation> dy_op;

    std::shared_ptr<viennacl::generator::symbolic_vector    <0, value_type>> dz_d;
    std::shared_ptr<viennacl::generator::symbolic_vector    <1, value_type>> dz_x;
    std::shared_ptr<viennacl::generator::symbolic_vector    <2, value_type>> dz_y;
    std::shared_ptr<viennacl::generator::symbolic_vector    <3, value_type>> dz_z;
    std::shared_ptr<viennacl::generator::cpu_symbolic_scalar<4, value_type>> dz_b;
    std::shared_ptr<viennacl::generator::custom_operation> dz_op;

    sys_func( const viennacl::vector<value_type> &_R ) : R( _R ),
	dx_d (new viennacl::generator::symbolic_vector    <0, value_type>()),
	dx_x (new viennacl::generator::symbolic_vector    <1, value_type>()),
	dx_y (new viennacl::generator::symbolic_vector    <2, value_type>()),
	dx_s (new viennacl::generator::cpu_symbolic_scalar<3, value_type>()),
	dx_op(new viennacl::generator::custom_operation(
		    (*dx_d) = (*dx_s) * ((*dx_y) - (*dx_x)))),
	dy_d (new viennacl::generator::symbolic_vector<0, value_type>()),
	dy_x (new viennacl::generator::symbolic_vector<1, value_type>()),
	dy_y (new viennacl::generator::symbolic_vector<2, value_type>()),
	dy_z (new viennacl::generator::symbolic_vector<3, value_type>()),
	dy_r (new viennacl::generator::symbolic_vector<4, value_type>()),
	dy_op(new viennacl::generator::custom_operation(
		    (*dy_d) = (*dy_r) * (*dy_x) - (*dy_y) - (*dy_x) * (*dy_z))),
	dz_d (new viennacl::generator::symbolic_vector    <0, value_type>()),
	dz_x (new viennacl::generator::symbolic_vector    <1, value_type>()),
	dz_y (new viennacl::generator::symbolic_vector    <2, value_type>()),
	dz_z (new viennacl::generator::symbolic_vector    <3, value_type>()),
	dz_b (new viennacl::generator::cpu_symbolic_scalar<4, value_type>()),
	dz_op(new viennacl::generator::custom_operation(
		    (*dz_d) = (*dz_x) * (*dz_y) - (*dz_b) * (*dz_z)))
    { }

    void operator()( const state_type &x , state_type &dxdt , value_type t ) const
    {
	viennacl::vector< value_type > dX = fusion::at_c<0>( dxdt );
	viennacl::vector< value_type > dY = fusion::at_c<1>( dxdt );
	viennacl::vector< value_type > dZ = fusion::at_c<2>( dxdt );

        viennacl::vector< value_type > &X = const_cast<viennacl::vector<value_type>&>(
		fusion::at_c< 0 >( x ));
        viennacl::vector< value_type > &Y = const_cast<viennacl::vector<value_type>&>(
		fusion::at_c< 1 >( x ));
        viennacl::vector< value_type > &Z = const_cast<viennacl::vector<value_type>&>(
		fusion::at_c< 1 >( x ));

	/*
	dX = -sigma * ( X - Y ) ;
	dY = R * X - Y - X * Z ;
	dZ = - b * Z + X * Y ;
	*/
	viennacl::ocl::enqueue( (*dx_op)(dX, X, Y, sigma) );
	viennacl::ocl::enqueue( (*dy_op)(dY, X, Y, Z, R) );
	viennacl::ocl::enqueue( (*dz_op)(dZ, X, Y, Z, b) );
    }
};

size_t n;
const value_type dt = 0.01;
const value_type t_max = 100.0;

int main( int argc , char **argv )
{
    using namespace std;

    n = argc > 1 ? atoi( argv[1] ) : 1024;


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
	    odeint::fusion_algebra , odeint::default_operations
	    > stepper;

    odeint::integrate_const( stepper , sys_func( R ) , S , value_type(0.0) , t_max , dt );

    std::vector< value_type > res( n );
    viennacl::copy( X , res );
    /*
    for( size_t i=0 ; i<n ; ++i )
     	cout << res[i] << "\t" << r[i] << "\n";
    */
    cout << res[0] << endl;

}
