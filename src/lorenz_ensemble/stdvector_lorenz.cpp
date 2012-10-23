#include <iostream>
#include <vector>
#include <utility>
#include <tuple>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/algebra/array_algebra.hpp>

namespace odeint = boost::numeric::odeint;

typedef boost::array<double, 3> state_type;
const double sigma = 10.0;
const double b = 8.0 / 3.0;

struct sys_func {
    double R;

    sys_func( const double &_R ) : R( _R ) { }

    void operator()( const state_type &x , state_type &dxdt , double t ) {
	dxdt[0] = sigma * (x[1] - x[0]);
	dxdt[1] = R * x[0] - x[1] - x[0] * x[2];
	dxdt[2] = x[0] * x[1] - b * x[2];
    }
};

size_t n;
const double dt = 0.01;
const double t_max = 100.0;

int main( int argc , char **argv )
{
    n = argc > 1 ? atoi(argv[1]) : 1024;
    using namespace std;

    std::vector<state_type> X(n);
    for (auto &x : X) x[0] = x[1] = x[2] = 10.0;

    std::vector<sys_func> sys;
    sys.reserve(n);
    double Rmin = 0.1 , Rmax = 50.0 , dR = ( Rmax - Rmin ) / double( n - 1 );
    for( size_t i=0 ; i<n ; ++i ) sys.emplace_back(Rmin + dR * double( i ));

    odeint::runge_kutta4<
	    state_type , double , state_type , double ,
	    odeint::array_algebra , odeint::default_operations
	    > stepper;

    for(double t = 0; t < t_max; t += dt)
#pragma omp parallel for private(stepper)
	for(int i = 0; i < n; i++)
	    stepper.do_step(std::ref(sys[i]), X[i], t, dt);

    cout << X[0][0] << " " << X[0][1] << " " << X[0][2] << endl;

}
