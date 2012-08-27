#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <random>
#include <algorithm>
#include <functional>
#include <CL/cl.hpp>

#include <viennacl/vector.hpp>
#include <viennacl/scalar.hpp>


#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/fusion_algebra.hpp>

#include <boost/fusion/sequence/intrinsic/at_c.hpp>

namespace odeint = boost::numeric::odeint;
namespace fusion = boost::fusion;

typedef double value_type;

typedef fusion::vector< viennacl::vector< value_type > , viennacl::vector< value_type > > state_type;

struct oscillator
{
    value_type m_omega;
    value_type m_amp;
    value_type m_offset;
    value_type m_omega_d;

    oscillator( value_type omega = 1.0 , value_type amp = 0.5 , value_type offset = 0.0 , value_type omega_d = 1.2 )
        : m_omega( omega ) , m_amp( amp ) , m_offset( offset ) , m_omega_d( omega_d ) { }

    void operator()( const state_type &x , state_type &dxdt , value_type t ) const
    {
        viennacl::vector< value_type > &dX = fusion::at_c< 0 >( dxdt );
        viennacl::vector< value_type > &dY = fusion::at_c< 1 >( dxdt );

        const viennacl::vector< value_type > &X = fusion::at_c< 0 >( x );
        const viennacl::vector< value_type > &Y = fusion::at_c< 1 >( x );

        value_type eps = m_offset + m_amp * cos( m_omega_d * t );
        dX =  m_omega * Y + eps * Y;
        dY = -m_omega * X + eps * Y;
    }
};

size_t N;
const value_type dt = 0.01;
const value_type t_max = 100.0;

int main( int argc , char **argv )
{
    using namespace std;

    N = argc > 1 ? atoi( argv[1] ) : 1024;

    // Get first available GPU device which supports double precision.
    std::vector<cl::Platform> platform;
    cl::Platform::get(&platform);
    if (platform.empty()) {
	std::cerr << "OpenCL platforms not found." << std::endl;
	return 1;
    }
    cl::Context context;
    std::vector<cl::Device> device;
    for(auto p = platform.begin(); device.empty() && p != platform.end(); p++) {
	std::vector<cl::Device> pldev;
	try {
	    p->getDevices(CL_DEVICE_TYPE_GPU, &pldev);
	    for(auto d = pldev.begin(); device.empty() && d != pldev.end(); d++) {
		if (!d->getInfo<CL_DEVICE_AVAILABLE>()) continue;
		std::string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();
		if (
			ext.find("cl_khr_fp64") == std::string::npos &&
			ext.find("cl_amd_fp64") == std::string::npos
		   ) continue;

		device.push_back(*d);
		context = cl::Context(device);
	    }
	} catch(...) {
	    device.clear();
	}
    }
    if (device.empty()) {
	std::cerr << "GPUs with double precision not found." << std::endl;
	return 1;
    }
    cl::CommandQueue queue(context, device[0]);
    std::vector<cl_device_id> dev_id(1, device[0]());
    std::vector<cl_command_queue> queue_id(1, queue());
    viennacl::ocl::setup_context(0, context(), dev_id, queue_id);

    cout << viennacl::ocl::current_device().name() << endl;

    std::mt19937 rng;
    std::normal_distribution< value_type > gauss( 0.0 , 1.0 );
    std::vector<value_type> x( N ) , y( N );
    std::generate( x.begin() , x.end() , std::bind( gauss , std::ref( rng ) ) );
    std::generate( y.begin() , y.end() , std::bind( gauss , std::ref( rng ) ) );


    state_type X;
    fusion::at_c< 0 >( X ).resize( N );
    fusion::at_c< 1 >( X ).resize( N );
    viennacl::copy( x , fusion::at_c< 0 >( X ) );
    viennacl::copy( y , fusion::at_c< 1 >( X ) );

    odeint::runge_kutta4<
        state_type , value_type , state_type , value_type ,
        odeint::fusion_algebra , odeint::default_operations
        > stepper;

    odeint::integrate_const( stepper , oscillator( 1.0 , 0.2 , 0.0 , 1.2 ) , X , value_type(0.0) , t_max , dt );

    std::vector< value_type > res( N );
    viennacl::copy( fusion::at_c< 0 >( X ) , res );
//     for( size_t i=0 ; i<n ; ++i )
//      	cout << res[i] << "\t" << r[i] << "\n";
    cout << res[0] << endl;

}
