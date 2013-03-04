#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <random>
#include <algorithm>

#include <vexcl/vexcl.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/vector_space_algebra.hpp>
#include <boost/numeric/odeint/external/vexcl/vexcl_resize.hpp>

namespace odeint = boost::numeric::odeint;

typedef double value_type;
typedef vex::vector< value_type > state_type;

using namespace std;

VEX_FUNCTION(pow3, value_type(value_type),  "return prm1 * prm1 * prm1;");

struct ham_lattice {
    ham_lattice(value_type beta, const vex::SpMat<double> &A) : beta(beta), A(A) { }

    void operator()(const state_type &q, state_type &dp) const {
        dp = (-beta) * pow3(q) + A * q;
    }

    value_type beta;
    const vex::SpMat< double > &A;
};

struct index_modulus {
    int N;

    index_modulus(int n) : N(n) {}

    inline int operator()(int idx) const {
	if( idx <  0 ) return idx + N;
	if( idx >= N ) return idx - N;
	return idx;
    }
};


int main( int argc , char **argv )
{
    size_t n1 = argc > 1 ? atoi(argv[1]) : 64;
    size_t n2 = n1;

    size_t n = n1 * n2;
    value_type K = 0.1;
    value_type beta = 0.01;
    value_type t_max = 100.0;
    value_type dt = 0.01;

    vex::Context ctx( vex::Filter::Exclusive( vex::Filter::Env && vex::Filter::DoublePrecision ) );
    std::cout << ctx << std::endl;

    std::vector<value_type> disorder( n );
    std::generate(disorder.begin(), disorder.end(), drand48);

    std::vector< double > val;
    std::vector< size_t > col;
    std::vector< size_t > row;

    size_t N = n1 * n2;

    val.reserve(N * 5);
    col.reserve(N * 5);
    row.reserve(N + 1);

    index_modulus index(N);

    row.push_back( 0 );
    for( int i=0 ; i < n1 ; ++i ) {
	for( int j=0 ; j < n2 ; ++j ) {
	    row.push_back( row.back() + 5 );
	    int idx = i * n2 + j;
	    int is[5] = { idx , index( idx + 1 ) , index( idx - 1 ) , index( idx - n2 ) , index( idx + n2 ) };
	    std::sort( is , is + 5 );
	    for( int i=0 ; i < 5 ; ++i ) {
		col.push_back( is[i] );
		if( is[i] == idx ) val.push_back( - disorder[idx]  - 4.0 * K );
		else val.push_back( K );
	    }
	}
    }

    vex::SpMat<double> A(ctx.queue(), N, N, row.data(), col.data(), val.data());

    std::pair< state_type , state_type > X( state_type( ctx.queue() , n1 * n2 ) , state_type( ctx.queue() , n1 * n2 ) );
    X.first = 0.0;
    X.second = 0.0;
    X.first[ n1/2*n2+n2/2 ] = 1.0;




    odeint::symplectic_rkn_sb3a_mclachlan<
        state_type , state_type , value_type , state_type , state_type , value_type ,
        odeint::vector_space_algebra , odeint::default_operations
        > stepper;

    ham_lattice sys( beta, A );
    odeint::integrate_const( stepper , std::ref( sys ) , X , value_type(0.0) , t_max , dt );


    std::vector< value_type > x1( n ) , p1( n );
    vex::copy( X.first , x1 );
    vex::copy( X.second , p1 );
    cout << x1[0] << "\t" << p1[0] << std::endl;
    /*
    for( size_t i=0 ; i<n1 ; ++i )
    {
        for( size_t j=0 ; j<n2 ; ++j )
            cout << i << "\t" << j << "\t" << x1[i*n2+j] << "\t" << p1[i*n2+j] << "\n";
        cout << "\n";
    }
    */
}
