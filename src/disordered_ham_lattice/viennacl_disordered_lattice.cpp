#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <tuple>
#include <random>
#include <algorithm>

#include <vexcl/vexcl.hpp>
#include <viennacl/vector.hpp>
#include "viennacl/ell_matrix.hpp"
#include "viennacl/linalg/prod.hpp"

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/vector_space_algebra.hpp>

#include <boost/numeric/odeint/external/viennacl/viennacl_operations.hpp>
#include <boost/numeric/odeint/external/viennacl/viennacl_resize.hpp>

namespace odeint = boost::numeric::odeint;

typedef double value_type;
typedef viennacl::vector< value_type > state_type;
typedef viennacl::ell_matrix< value_type > matrix_type; 

using namespace std;

struct ham_lattice
{
    ham_lattice( long n1 , long n2 , value_type K ,
	    const std::vector< value_type > &disorder ) : m_N( n1 * n2 )
    {
	if( disorder.size() != static_cast<size_t>(n1 * n2) ) throw ;


	std::vector<std::map<unsigned int, value_type>> cpu_matrix(m_N);

	for( long i = 0, idx = 0 ; i < n1 ; ++i ) 
	{
	    for( long j = 0 ; j < n2 ; ++j, ++idx )
	    {
		cpu_matrix[idx][index_modulus(idx - n2)] = K;
		cpu_matrix[idx][index_modulus(idx - 1)] = K;
		cpu_matrix[idx][idx] = - disorder[idx]  - 4.0 * K;
		cpu_matrix[idx][index_modulus(idx + 1)] = K;
		cpu_matrix[idx][index_modulus(idx + n2)] = K;
	    }
	}

	m_A.reset( new matrix_type( m_N , m_N) );

	copy(viennacl::tools::const_sparse_matrix_adapter<double>(
		    cpu_matrix, m_N, m_N), *m_A);

    }

    inline long index_modulus( long idx )
    {
        if( idx < 0 ) return idx + m_N;
        if( idx >= m_N ) return idx - m_N;
        return idx;
    }

    void operator()( const state_type &q , state_type &dp ) const
    {
        dp = viennacl::linalg::prod(*m_A, q);
    }

    long m_N ;
    std::shared_ptr< matrix_type > m_A;
};


int main( int argc , char **argv )
{
    size_t n1 = argc > 1 ? atoi(argv[1]) : 64;
    size_t n2 = n1;

    size_t n = n1 * n2;
    value_type K = 0.1;
    value_type t_max = 1000.0;
    value_type dt = 0.01;
    
    vex::Context ctx( vex::Filter::Env && vex::Filter::DoublePrecision && vex::Filter::Count(1));
    std::vector<cl_device_id> dev_id(1, ctx.queue(0).getInfo<CL_QUEUE_DEVICE>()());
    std::vector<cl_command_queue> queue_id(1, ctx.queue(0)());
    viennacl::ocl::setup_context(0, ctx.context(0)(), dev_id, queue_id);
    std::cout << ctx << std::endl;
    

    std::vector<value_type> disorder( n, 0 );

    std::pair< state_type , state_type > X( state_type( n1 * n2 ) , state_type( n1 * n2 ) );
    viennacl::copy(disorder, X.first);
    viennacl::copy(disorder, X.second);
    X.first[ n1/2*n2+n2/2 ] = 1.0;

    std::mt19937 rng;
    std::uniform_real_distribution< value_type > dist( 0.0 , 1.0 );
    for( size_t i=0 ; i<n ; ++i ) disorder[i] = dist( rng );


    odeint::symplectic_rkn_sb3a_mclachlan<
        state_type , state_type , value_type , state_type , state_type , value_type ,
        odeint::vector_space_algebra , odeint::viennacl_operations
        > stepper;

    ham_lattice sys( n1 , n2 , K , disorder );
    odeint::integrate_const( stepper , std::ref( sys ) , X , value_type(0.0) , t_max , dt );


    std::vector< value_type > x1( n ) , p1( n );
    viennacl::copy( X.first , x1 );
    viennacl::copy( X.second , p1 );
    cout << x1[0] << "\t" << p1[0] << std::endl;
    /*
    for( size_t i=0 ; i<n1 ; ++i )
    {
        for( size_t j=0 ; j<n2 ; ++j )
            cout << i << "\t" << j << "\t" << x1[i*n2+j] << "\t" << p1[i*n2+j] << "\n";
        cout << "\n";
    }
    */

    exit(0);
}
