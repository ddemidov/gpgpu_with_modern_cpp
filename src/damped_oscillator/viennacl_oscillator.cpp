#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <random>
#include <algorithm>
#include <functional>
#include <memory>

#include <vexcl/vexcl.hpp>
#include <vexcl/exclusive.hpp>
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
    viennacl::vector< value_type >
    > state_type;

// namespace boost { namespace numeric { namespace odeint {

// template<>
// struct is_resizeable< state_type > : boost::true_type { };

// template<>
// struct resize_impl< state_type , state_type >
// {
//     static void resize( state_type &x1 , const state_type &x2 )
//     {
//         fusion::at_c< 0 >( x1 ).resize( fusion::at_c< 0 >( x2 ).size() , false );
//         fusion::at_c< 1 >( x1 ).resize( fusion::at_c< 1 >( x2 ).size() , false );
//     }
// };

// template<>
// struct same_size_impl< state_type , state_type >
// {
//     static bool same_size( const state_type &x1 , const state_type &x2 )
//     {
//         return
//             ( fusion::at_c< 0 >( x1 ).size() == fusion::at_c< 0 >( x2 ).size() ) &&
//             ( fusion::at_c< 1 >( x1 ).size() == fusion::at_c< 1 >( x2 ).size() ) ;
//     }
// };

// } } }


struct oscillator
{
    value_type m_omega;
    value_type m_amp;
    value_type m_offset;
    value_type m_omega_d;

    viennacl::generator::symbolic_vector<0, value_type> sym_dx;
    viennacl::generator::symbolic_vector<1, value_type> sym_dy;
    viennacl::generator::symbolic_vector<2, value_type> sym_x;
    viennacl::generator::symbolic_vector<3, value_type> sym_y;
    viennacl::generator::cpu_symbolic_scalar<4, value_type> sym_eps;
    viennacl::generator::cpu_symbolic_scalar<5, value_type> sym_omega;

    viennacl::generator::custom_operation op;

    oscillator(value_type omega, value_type amp, value_type offset, value_type omega_d)
        : m_omega(omega), m_amp(amp) , m_offset(offset) , m_omega_d(omega_d),
	  op(sym_dx = sym_eps * sym_x + sym_omega * sym_y,
	     sym_dy = sym_eps * sym_y - sym_omega * sym_x,
	     "oscillator"
	    )
    {
    }

    void operator()( const state_type &x , state_type &dxdt , value_type t )
    {
        viennacl::vector< value_type > &X = const_cast<viennacl::vector<value_type>&>(fusion::at_c< 0 >( x ));
        viennacl::vector< value_type > &Y = const_cast<viennacl::vector<value_type>&>(fusion::at_c< 1 >( x ));

        viennacl::vector< value_type > &dX = fusion::at_c< 0 >( dxdt );
        viennacl::vector< value_type > &dY = fusion::at_c< 1 >( dxdt );

        value_type eps = m_offset + m_amp * cos( m_omega_d * t );

	viennacl::ocl::enqueue( op(dX, dY, X, Y, eps, m_omega) );
    }
};

size_t N;
const value_type dt = 0.01;
const value_type t_max = 100.0;

int main( int argc , char **argv )
{
    using namespace std;

    N = argc > 1 ? atoi( argv[1] ) : 1024;

    vex::Context ctx( vex::Filter::Exclusive( vex::Filter::Env && vex::Filter::Count(1)) );

    std::vector<cl_device_id> dev_id(1, ctx.queue(0).getInfo<CL_QUEUE_DEVICE>()());
    std::vector<cl_command_queue> queue_id(1, ctx.queue(0)());

    viennacl::ocl::setup_context(0, ctx.context(0)(), dev_id, queue_id);

    cout << ctx << endl;

    std::vector<value_type> x( 2 * N );
    std::generate( x.begin() , x.end() , drand48 );


    state_type S;

    viennacl::vector<value_type> &X = fusion::at_c<0>(S);
    viennacl::vector<value_type> &Y = fusion::at_c<1>(S);

    X.resize( N );
    Y.resize( N );

    viennacl::copy( x.begin(), x.begin() + N, X.begin() );
    viennacl::copy( x.begin() + N, x.end(), Y.begin() );

    odeint::runge_kutta4<
        state_type , value_type , state_type , value_type ,
        odeint::fusion_algebra , odeint::viennacl_operations
        > stepper;

    oscillator sys(1.0, 0.2, 0.0, 1.2);
    odeint::integrate_const( stepper , std::ref( sys ), S , value_type(0.0) , t_max , dt );

    std::vector< value_type > res( N );
    viennacl::copy( X, res );
//     for( size_t i=0 ; i<n ; ++i )
//      	cout << res[i] << "\t" << r[i] << "\n";
    cout << res[0] << endl;

    exit(0);
}
