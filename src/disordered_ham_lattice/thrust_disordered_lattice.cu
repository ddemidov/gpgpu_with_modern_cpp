#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/thrust/thrust_algebra.hpp>
#include <boost/numeric/odeint/external/thrust/thrust_operations.hpp>
#include <boost/numeric/odeint/external/thrust/thrust_resize.hpp>

#include <cusparse_v2.h>

using namespace std;
namespace odeint = boost::numeric::odeint;

typedef double value_type;
typedef thrust::device_vector< value_type > state_type;

struct ham_lattice {
    value_type beta;
    cusparseHandle_t   handle;
    cusparseMatDescr_t descr;
    cusparseHybMat_t   A;

    ham_lattice( value_type beta, cusparseHandle_t handle,
	    cusparseMatDescr_t descr, cusparseHybMat_t A
	    ) : beta(beta) , descr(descr), handle(handle), A(A) { }

    struct scaled_pow3_functor {
        value_type beta;

        scaled_pow3_functor( value_type beta ) : beta(beta) {}

        __host__ __device__ value_type operator()( value_type q ) const {
	    return beta * q * q * q;
        }
    };

    void operator()( const state_type &q , state_type &dp ) const
    {
	static value_type one = 1;

	thrust::transform(q.begin(), q.end(), dp.begin(),
		scaled_pow3_functor(-beta));

	cusparseDhybmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&one, descr, A,
		thrust::raw_pointer_cast(&q[0]), &one,
		thrust::raw_pointer_cast(&dp[0])
		);
    }

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

    std::vector<value_type> disorder( n );
    std::generate( disorder.begin(), disorder.end(), drand48 );

    // Create CUSPARSE matrix.
    cusparseHandle_t   handle;
    cusparseMatDescr_t descr;
    cusparseHybMat_t   hyb;

    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseCreateHybMat(&hyb);

    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    {
	std::vector< double > val;
	std::vector< int > col;
	std::vector< int > row;

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

	thrust::device_vector<int>    dev_row(row);
	thrust::device_vector<int>    dev_col(col);
	thrust::device_vector<double> dev_val(val);

	cusparseDcsr2hyb(handle, N, N, descr,
		thrust::raw_pointer_cast(&dev_val[0]),
		thrust::raw_pointer_cast(&dev_row[0]),
		thrust::raw_pointer_cast(&dev_col[0]),
		hyb, 5, CUSPARSE_HYB_PARTITION_AUTO
		);
    }

    std::pair<state_type, state_type> X(
	    state_type( n1 * n2 ),
	    state_type( n1 * n2 )
	    );
    thrust::fill(X.first.begin(),  X.first.end(),  0);
    thrust::fill(X.second.begin(), X.second.end(), 0);
    X.first[ n1/2*n2+n2/2 ] = 1.0;


    odeint::symplectic_rkn_sb3a_mclachlan<
        state_type , state_type , value_type , state_type , state_type , value_type ,
        odeint::thrust_algebra , odeint::thrust_operations
        > stepper;

    odeint::integrate_const( stepper , ham_lattice(beta , handle, descr, hyb),
	    X, value_type(0.0), t_max, dt );


    std::vector< value_type > x1( n ) , p1( n );
    thrust::copy( X.first.begin(),  X.first.end(),  x1.begin() );
    thrust::copy( X.second.begin(), X.second.end(), p1.begin() );

    cout << x1[0] << "\t" << p1[0] << std::endl;
}
