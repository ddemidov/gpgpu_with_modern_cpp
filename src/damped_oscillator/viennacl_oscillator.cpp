#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <random>
#include <algorithm>
#include <functional>
#include <memory>

#include <vexcl/vexcl.hpp>
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
    }
};

template<>
struct same_size_impl< state_type , state_type >
{
    static bool same_size( const state_type &x1 , const state_type &x2 )
    {
        return
            ( fusion::at_c< 0 >( x1 ).size() == fusion::at_c< 0 >( x2 ).size() ) &&
            ( fusion::at_c< 1 >( x1 ).size() == fusion::at_c< 1 >( x2 ).size() ) ;
    }
};

} } }

struct oscillator
{
    value_type m_omega;
    value_type m_amp;
    value_type m_offset;
    value_type m_omega_d;

    std::shared_ptr<viennacl::generator::symbolic_vector    <0, value_type>> sym_r;
    std::shared_ptr<viennacl::generator::symbolic_vector    <1, value_type>> sym_x;
    std::shared_ptr<viennacl::generator::symbolic_vector    <2, value_type>> sym_y;
    std::shared_ptr<viennacl::generator::cpu_symbolic_scalar<3, value_type>> sym_a;
    std::shared_ptr<viennacl::generator::cpu_symbolic_scalar<4, value_type>> sym_b;
    std::shared_ptr<viennacl::generator::custom_operation> ax_plus_by;

    oscillator(value_type omega, value_type amp, value_type offset, value_type omega_d)
        : m_omega(omega), m_amp(amp) , m_offset(offset) , m_omega_d(omega_d),
	  sym_r(new viennacl::generator::symbolic_vector    <0, value_type>()),
	  sym_x(new viennacl::generator::symbolic_vector    <1, value_type>()),
	  sym_y(new viennacl::generator::symbolic_vector    <2, value_type>()),
	  sym_a(new viennacl::generator::cpu_symbolic_scalar<3, value_type>()),
	  sym_b(new viennacl::generator::cpu_symbolic_scalar<4, value_type>()),
	  ax_plus_by(new viennacl::generator::custom_operation(
		      (*sym_r) = (*sym_a) * (*sym_x) + (*sym_b) * (*sym_y)
		      ))
    { }

    void operator()( const state_type &x , state_type &dxdt , value_type t )
    {
        viennacl::vector< value_type > &dX = fusion::at_c< 0 >( dxdt );
        viennacl::vector< value_type > &dY = fusion::at_c< 1 >( dxdt );

        viennacl::vector< value_type > &X = const_cast<viennacl::vector<value_type>&>(
		fusion::at_c< 0 >( x ));
        viennacl::vector< value_type > &Y = const_cast<viennacl::vector<value_type>&>(
		fusion::at_c< 1 >( x ));

        value_type eps = m_offset + m_amp * cos( m_omega_d * t );
	value_type minus_omega = -m_omega;
	viennacl::ocl::enqueue( (*ax_plus_by)(dX, Y, X, m_omega, eps) );
	viennacl::ocl::enqueue( (*ax_plus_by)(dY, X, Y, minus_omega, eps) );
    }
};

size_t N;
const value_type dt = 0.01;
const value_type t_max = 100.0;

int main( int argc , char **argv )
{
    using namespace std;

    N = argc > 1 ? atoi( argv[1] ) : 1024;

    vex::Context ctx( vex::Filter::Env && vex::Filter::Count(1));

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
        odeint::fusion_algebra , odeint::default_operations
        > stepper;

    odeint::integrate_const( stepper , oscillator(1.0, 0.2, 0.0, 1.2),
	    S , value_type(0.0) , t_max , dt );

    std::vector< value_type > res( N );
    viennacl::copy( X, res );
//     for( size_t i=0 ; i<n ; ++i )
//      	cout << res[i] << "\t" << r[i] << "\n";
    cout << res[0] << endl;

}
