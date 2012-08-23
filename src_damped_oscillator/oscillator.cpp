#include <iostream>
#include <array>

#include <boost/numeric/odeint.hpp>

typedef std::array< double , 2 > state_type;

struct oscillator
{
    double m_omega;
    double m_amp;
    double m_offset;
    double m_omega_d;

    oscillator( double omega = 1.0 , double amp = 0.5 , double offset = 0.0 , double omega_d = 1.2 )
        : m_omega( omega ) , m_amp( amp ) , m_offset( offset ) , m_omega_d( omega_d ) { }

    void operator()( const state_type &x , state_type &dxdt , double t ) const
    {
        double eps = m_offset + m_amp * cos( m_omega_d * t );
        dxdt[0] =  m_omega * x[1] + eps * x[0];
        dxdt[1] = -m_omega * x[0] + eps * x[1];
    }
};


int main( int argc , char **argv )
{
    using namespace boost::numeric::odeint;

    state_type x = {{ 1.0 , 0.0 }};
    integrate_const( runge_kutta4< state_type >() , 
                     oscillator( 1.0 , 0.2 , 0.0 , 1.2 ) , x , 0.0 , 100.0 , 0.1 ,
                     []( const state_type &x , double t ) {
                         cout << t << "\t" << x[0] << "\t" << x[1] << "\n"; } );
    
    return 0;
}
