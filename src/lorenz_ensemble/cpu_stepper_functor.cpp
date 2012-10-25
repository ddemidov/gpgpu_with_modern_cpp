#include <iostream>
#include <vector>

#include <boost/numeric/odeint.hpp>
#include <boost/iterator/zip_iterator.hpp>

namespace odeint = boost::numeric::odeint;

//---------------------------------------------------------------------------
template <typename T>
struct point3d {
    T x;
    T y;
    T z;
};

template <typename T>
inline point3d<T> operator+(point3d<T> a, point3d<T> b) {
    point3d<T> c = {a.x + b.x, a.y + b.y, a.z + b.z};
    return c;
}

template <typename T>
inline point3d<T> operator-(point3d<T> a, point3d<T> b) {
    point3d<T> c = {a.x - b.x, a.y - b.y, a.z - b.z};
    return c;
}

template <typename T>
inline point3d<T> operator*(T a, point3d<T> b) {
    point3d<T> c = {a * b.x, a * b.y, a * b.z};
    return c;
}


//---------------------------------------------------------------------------
typedef double value_type;
typedef point3d<value_type> state_type;

const value_type sigma = 10.0;
const value_type b = 8.0 / 3.0;
const value_type dt = 0.01;
const value_type t_max = 100.0;

struct lorenz_system {
    value_type R;
    
    lorenz_system(value_type r) : R(r) {}

    inline void operator()(const state_type &s, state_type &dsdt, value_type t) {
	dsdt.x = sigma * (s.y - s.x);
	dsdt.y = R * s.x - s.y - s.x * s.z;
	dsdt.z = s.x * s.y - b * s.z;
    }
};

struct stepper_functor {
    odeint::runge_kutta4<
	    state_type, value_type, state_type, value_type,
	    odeint::vector_space_algebra, odeint::default_operations
	    > stepper;

    value_type t;

    template <class T>
    void operator()(T s) {
	auto &state = boost::get<0>(s);
	auto &sys   = boost::get<1>(s);
	stepper.do_step(sys, state, t, dt);
    }
};

//---------------------------------------------------------------------------
std::ostream& operator<<(std::ostream &os, state_type s) {
    return os << "[" << s.x << " " << s.y << " " << s.z << "]";
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    size_t n = argc > 1 ? atoi(argv[1]) : 1024;

    std::vector<lorenz_system> ensemble;
    ensemble.reserve(n);

    value_type Rmin = 0.1 , Rmax = 50.0 , dR = ( Rmax - Rmin ) / value_type( n - 1 );
    for( size_t i=0 ; i<n ; ++i )
	ensemble.emplace_back(Rmin + dR * value_type( i ));

    std::vector<state_type> x(n);
    std::for_each(x.begin(), x.end(), [](state_type &s) { s.x = s.y = s.z = 10.0; });

    stepper_functor step;
    for(step.t = 0; step.t < t_max; step.t += dt)
	std::for_each(
		boost::make_zip_iterator(
		    boost::make_tuple(x.begin(), ensemble.begin())),
		boost::make_zip_iterator(
		    boost::make_tuple(x.end(),   ensemble.end())),
		step);

    std::cout << x[0] << std::endl;
}
