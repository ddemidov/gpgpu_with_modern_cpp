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

struct ham_lattice
{
    ham_lattice( const std::vector< cl::CommandQueue > &queue , size_t n1 , size_t n2 , value_type K , std::vector< double > disorder )
        : m_n1( 0 ) , m_n2( 0 ) , m_N( 0 ) , m_K( 0.0 ) , m_A()
    {
        init( queue , n1 , n2 , K , disorder );
    }

    inline size_t index_modulus( long idx )
    {
        if( idx < 0 ) return size_t( idx + m_N );
        if( idx >= m_N ) return size_t( idx - m_N );
        return size_t( idx );
    }

    void init( const std::vector< cl::CommandQueue > &queue , size_t n1 , size_t n2 , value_type K , std::vector< double > disorder )
    {
        m_n1 = n1;
        m_n2 = n2;
        m_N = m_n1 * m_n2;
        m_K = K;

        if( disorder.size() != n1 * n2 ) throw ;

        std::vector< double > val;
        std::vector< size_t > col;
        std::vector< size_t > row;

	val.reserve(m_N * 5);
	col.reserve(m_N * 5);
	row.reserve(m_N + 1);

        row.push_back( 0 );
        for( size_t i=0 ; i<m_n1 ; ++i ) 
        {
            for( size_t j=0 ; j<m_n2 ; ++j )
            {
                row.push_back( row.back() + 5 );
                size_t idx = i * m_n2 + j;
                long ii1 = idx + 1, ii2 = idx - 1 , ii3 = idx - m_n2 , ii4 = idx + m_n2;
                std::array< size_t , 5 > is = {{ idx , index_modulus( ii1 ) , index_modulus( ii2 ) , index_modulus( ii3 ) , index_modulus( ii4 ) }};
                sort( is.begin() , is.end() );
                for( size_t i=0 ; i<is.size() ; ++i )
                {
                    col.push_back( is[i] );
                    if( is[i] == idx ) val.push_back( - disorder[idx]  - 4.0 * K );
                    else val.push_back( K );
                }
            }
        }

        m_A.reset( new vex::SpMat< double >( queue , m_N , m_N , row.data() , col.data() , val.data() ) );
    }

    void operator()( const state_type &q , state_type &dp ) const
    {
        dp = (*m_A) * q;
    }

    size_t m_n1 , m_n2 , m_N ;
    value_type m_K;
    std::unique_ptr< vex::SpMat< double > > m_A;
};


int main( int argc , char **argv )
{
    size_t n1 = 64 , n2 = 64;
    size_t n = n1 * n2;
    value_type K = 0.1;
    value_type t_max = 1000.0;
    value_type dt = 0.01;
    
    vex::Context ctx( vex::Filter::Env && vex::Filter::DoublePrecision );
    std::cout << ctx << std::endl;
    

    std::vector<value_type> disorder( n );
    std::mt19937 rng;
    std::uniform_real_distribution< value_type > dist( 0.0 , 1.0 );
    for( size_t i=0 ; i<n ; ++i ) disorder[i] = dist( rng );

    std::pair< state_type , state_type > X( state_type( ctx.queue() , n1 * n2 ) , state_type( ctx.queue() , n1 * n2 ) );
    X.first = 0.0;
    X.second = 0.0;
    X.first[ n1/2*n2+n2/2 ] = 1.0;




    odeint::symplectic_rkn_sb3a_mclachlan<
        state_type , state_type , value_type , state_type , state_type , value_type ,
        odeint::vector_space_algebra , odeint::default_operations
        > stepper;

    ham_lattice sys( ctx.queue() , n1 , n2 , K , disorder );
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
