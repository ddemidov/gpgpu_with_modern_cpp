#include <iostream>
#include <vector>
#include <utility>
#include <tuple>

#include <viennacl/vector.hpp>
#include <viennacl/vector_proxy.hpp>


#include <boost/numeric/odeint.hpp>


namespace odeint = boost::numeric::odeint;

typedef double value_type;

typedef viennacl::vector< value_type > state_type;

namespace boost { namespace numeric { namespace odeint {

template<>
struct is_resizeable< state_type > : boost::true_type { };

template<>
struct resize_impl< state_type , state_type >
{
    static void resize( state_type &x1 , const state_type &x2 )
    {
	x1.resize( x2.size() , false );
    }
};

template<>
struct same_size_impl< state_type , state_type >
{
    static bool same_size( const state_type &x1 , const state_type &x2 )
    {
	return x1.size() == x2.size();
    }
};


} } }


const value_type sigma = 10.0;
const value_type b = 8.0 / 3.0;
const value_type R = 28.0;

struct sys_func
{
    const state_type &R;
    size_t n;

    sys_func( const state_type &_R ) : R( _R ) , n( _R.size() ) { }

    void operator()( const state_type &x , state_type &dxdt , value_type t ) const
    {
	viennacl::vector_range< state_type > dX( dxdt , viennacl::range( 0 , n ) );
	viennacl::vector_range< state_type > dY( dxdt , viennacl::range( n , 2 * n ) );
	viennacl::vector_range< state_type > dZ( dxdt , viennacl::range( 2 * n , 3 * n ) );

	viennacl::vector_range< const state_type > X( x , viennacl::range( 0 , n ) );
	viennacl::vector_range< const state_type > Y( x , viennacl::range( n , 2 * n ) );
	viennacl::vector_range< const state_type > Z( x , viennacl::range( 2 * n , 3 * n ) );

	dX = -sigma * ( X - Y ) ;
	dY = R * X - Y - X * Z ;
	dZ = - b * Z + X * Y ;
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
    for( size_t i=0 ; i<n ; ++i ) r[i] = Rmin + dR * value_type( i );

    std::vector< value_type > x( 3*n , 10 );
    state_type X( 3 * n );
    viennacl::copy( x , X );

    state_type R( n );
    viennacl::copy( r , R );

    odeint::runge_kutta4<
	    state_type , value_type , state_type , value_type ,
	    odeint::vector_space_algebra , odeint::default_operations
	    > stepper;

    odeint::integrate_const( stepper , sys_func( R ) , X , value_type(0.0) , t_max , dt );

    std::vector< value_type > res( n );
    viennacl::copy( X.begin() , X.begin() + n , res.begin() );
    for( size_t i=0 ; i<n ; ++i )
     	cout << res[i] << "\t" << r[i] << "\n";
//    cout << res[0] << endl;

}
