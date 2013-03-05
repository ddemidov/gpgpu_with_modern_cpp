#include <iostream>
#include <vector>
#include <algorithm>

#include <vexcl/devlist.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/util/is_resizeable.hpp>
#include <boost/numeric/odeint/util/resize.hpp>
#include <boost/numeric/odeint/util/same_size.hpp>

namespace odeint = boost::numeric::odeint;
typedef double value_type;

static const size_t wgsize = 256;

static cl::Context      ctx;
static cl::Device       device;
static cl::CommandQueue queue;
static cl::Program      program;

const char clbuf_source[] =
"#if defined(cl_khr_fp64)\n"
"#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
"#elif defined(cl_amd_fp64)\n"
"#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
"#endif\n"
"\n"
"typedef double real;\n"
"\n"
"kernel void scale_sum2(\n"
"    ulong n,\n"
"    global real *v0, \n"
"    global real *v1, \n"
"    global real *v2, \n"
"    real a1,\n"
"    real a2\n"
"    )\n"
"{\n"
"    for(size_t i = get_global_id(0); i < n; i += get_global_size(0))\n"
"        v0[i] =\n"
"            a1 * v1[i] +\n"
"            a2 * v2[i];\n"
"}\n"
"\n"
"kernel void scale_sum3(\n"
"    ulong n,\n"
"    global real *v0, \n"
"    global real *v1, \n"
"    global real *v2, \n"
"    global real *v3, \n"
"    real a1,\n"
"    real a2,\n"
"    real a3\n"
"    )\n"
"{\n"
"    for(size_t i = get_global_id(0); i < n; i += get_global_size(0))\n"
"        v0[i] =\n"
"            a1 * v1[i] +\n"
"            a2 * v2[i] +\n"
"            a3 * v3[i];\n"
"}\n"
"\n"
"kernel void scale_sum4(\n"
"    ulong n,\n"
"    global real *v0, \n"
"    global real *v1, \n"
"    global real *v2, \n"
"    global real *v3, \n"
"    global real *v4, \n"
"    real a1,\n"
"    real a2,\n"
"    real a3,\n"
"    real a4\n"
"    )\n"
"{\n"
"    for(size_t i = get_global_id(0); i < n; i += get_global_size(0))\n"
"        v0[i] =\n"
"            a1 * v1[i] +\n"
"            a2 * v2[i] +\n"
"            a3 * v3[i] +\n"
"            a4 * v4[i];\n"
"}\n"
"\n"
"kernel void scale_sum5(\n"
"    ulong n,\n"
"    global real *v0, \n"
"    global real *v1, \n"
"    global real *v2, \n"
"    global real *v3, \n"
"    global real *v4, \n"
"    global real *v5, \n"
"    real a1,\n"
"    real a2,\n"
"    real a3,\n"
"    real a4,\n"
"    real a5\n"
"    )\n"
"{\n"
"    for(size_t i = get_global_id(0); i < n; i += get_global_size(0))\n"
"        v0[i] =\n"
"            a1 * v1[i] +\n"
"            a2 * v2[i] +\n"
"            a3 * v3[i] +\n"
"            a4 * v4[i] +\n"
"            a5 * v5[i];\n"
"}\n"
"\n"
"kernel void oscillator_system(\n"
"    ulong n,\n"
"    global real *dsdt,\n"
"    global real *s,\n"
"    global real *omega\n"
"    )\n"
"{\n"
"    for(size_t i = get_global_id(0); i < n; i += get_global_size(0)) {\n"
"        real xl = s[i > 0 ? i - 1 : 0];\n"
"        real x0 = s[i];\n"
"        real xr = s[i < n - 1 ? i + 1 : n - 1];\n"
"        dsdt[i] = omega[i] + sin(xl - x0) + sin(x0 - xr);\n"
"    }\n"
"}\n";

inline size_t alignup(size_t n, size_t m = 16U) {
    return n % m ? n - n % m + m : n;
}

template <typename T>
struct clbuf {
    size_t      n;
    cl::Buffer  data;

    clbuf() : n(0) {}

    clbuf(size_t n)
        : n(n), data(ctx, CL_MEM_READ_WRITE, sizeof(T) * n) 
    {}

    clbuf(const std::vector<T> &host)
        : n(host.size()),
          data(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                  sizeof(T) * host.size(), const_cast<T*>(host.data())) 
    {}

    void swap(clbuf &other) {
        std::swap(n,    other.n);
        std::swap(data, other.data);
    }
};

// Resizing
namespace boost { namespace numeric { namespace odeint {

template <typename T>
struct is_resizeable< clbuf<T> > : boost::true_type {};

template< typename T >
struct resize_impl< clbuf<T> , clbuf<T> >
{
    static void resize( clbuf<T> &x1 , const clbuf<T> &x2 )
    {
        clbuf<T>(x2.n).swap(x1);
    }
};

template< typename T >
struct same_size_impl< clbuf<T> , clbuf<T> >
{
    static bool same_size( const clbuf<T> &x1 , const clbuf<T> &x2 )
    {
        return x1.n == x2.n;
    }
};

} } }

// Operations
struct clbuf_operations {
    template< class Fac1 = double , class Fac2 = Fac1 >
    struct scale_sum2
    {
        const Fac1 m_alpha1;
        const Fac2 m_alpha2;

        scale_sum2( Fac1 alpha1 , Fac2 alpha2 )
            : m_alpha1( alpha1 ) , m_alpha2( alpha2 )
        { }

        template< class T1 , class T2 , class T3 >
        void operator()(clbuf<T1> &v1 ,
                  const clbuf<T2> &v2 ,
                  const clbuf<T3> &v3
                  ) const
        {
            static cl::Kernel krn(program, "scale_sum2");

            uint pos = 0;
            krn.setArg(pos++, v1.n);
            krn.setArg(pos++, v1.data);
            krn.setArg(pos++, v2.data);
            krn.setArg(pos++, v3.data);
            krn.setArg(pos++, m_alpha1);
            krn.setArg(pos++, m_alpha2);

            queue.enqueueNDRangeKernel(
                    krn, cl::NullRange, alignup(v1.n, wgsize), wgsize
                    );
        }

        typedef void result_type;
    };

    template< class Fac1 = double , class Fac2 = Fac1 , class Fac3 = Fac2 >
    struct scale_sum3
    {
        const Fac1 m_alpha1;
        const Fac2 m_alpha2;
        const Fac3 m_alpha3;

        scale_sum3( Fac1 alpha1 , Fac2 alpha2 , Fac3 alpha3 )
            : m_alpha1( alpha1 ) , m_alpha2( alpha2 ) , m_alpha3( alpha3 )
        { }

        template< class T1 , class T2 , class T3 , class T4 >
        void operator()(clbuf<T1> &v1 ,
                  const clbuf<T2> &v2 ,
                  const clbuf<T3> &v3 ,
                  const clbuf<T4> &v4
                  ) const
        {
            static cl::Kernel krn(program, "scale_sum3");

            uint pos = 0;
            krn.setArg(pos++, v1.n);
            krn.setArg(pos++, v1.data);
            krn.setArg(pos++, v2.data);
            krn.setArg(pos++, v3.data);
            krn.setArg(pos++, v4.data);
            krn.setArg(pos++, m_alpha1);
            krn.setArg(pos++, m_alpha2);
            krn.setArg(pos++, m_alpha3);

            queue.enqueueNDRangeKernel(
                    krn, cl::NullRange, alignup(v1.n, wgsize), wgsize
                    );
        }

        typedef void result_type;
    };


    template< class Fac1 = double , class Fac2 = Fac1 , class Fac3 = Fac2 , class Fac4 = Fac3 >
    struct scale_sum4
    {
        const Fac1 m_alpha1;
        const Fac2 m_alpha2;
        const Fac3 m_alpha3;
        const Fac4 m_alpha4;

        scale_sum4( Fac1 alpha1 , Fac2 alpha2 , Fac3 alpha3 , Fac4 alpha4 )
        : m_alpha1( alpha1 ) , m_alpha2( alpha2 ) , m_alpha3( alpha3 ) , m_alpha4( alpha4 ) { }

        template< class T1 , class T2 , class T3 , class T4 , class T5 >
        void operator()(clbuf<T1> &v1 ,
                  const clbuf<T2> &v2 ,
                  const clbuf<T3> &v3 ,
                  const clbuf<T4> &v4 ,
                  const clbuf<T5> &v5
                  ) const
        {
            static cl::Kernel krn(program, "scale_sum4");

            uint pos = 0;
            krn.setArg(pos++, v1.n);
            krn.setArg(pos++, v1.data);
            krn.setArg(pos++, v2.data);
            krn.setArg(pos++, v3.data);
            krn.setArg(pos++, v4.data);
            krn.setArg(pos++, v5.data);
            krn.setArg(pos++, m_alpha1);
            krn.setArg(pos++, m_alpha2);
            krn.setArg(pos++, m_alpha3);
            krn.setArg(pos++, m_alpha4);

            queue.enqueueNDRangeKernel(
                    krn, cl::NullRange, alignup(v1.n, wgsize), wgsize
                    );
        }

        typedef void result_type;
    };


    template< class Fac1 = double , class Fac2 = Fac1 , class Fac3 = Fac2 , class Fac4 = Fac3 , class Fac5 = Fac4 >
    struct scale_sum5
    {
        const Fac1 m_alpha1;
        const Fac2 m_alpha2;
        const Fac3 m_alpha3;
        const Fac4 m_alpha4;
        const Fac5 m_alpha5;

        scale_sum5( Fac1 alpha1 , Fac2 alpha2 , Fac3 alpha3 , Fac4 alpha4 , Fac5 alpha5 )
        : m_alpha1( alpha1 ) , m_alpha2( alpha2 ) , m_alpha3( alpha3 ) , m_alpha4( alpha4 ) , m_alpha5( alpha5 ) { }

        template< class T1 , class T2 , class T3 , class T4 , class T5 , class T6 >
        void operator()(clbuf<T1> &v1 ,
                const clbuf<T2> &v2 ,
                const clbuf<T3> &v3 ,
                const clbuf<T4> &v4 ,
                const clbuf<T5> &v5 ,
                const clbuf<T6> &v6
                ) const
        {

            static cl::Kernel krn(program, "scale_sum5");

            uint pos = 0;
            krn.setArg(pos++, v1.n);
            krn.setArg(pos++, v1.data);
            krn.setArg(pos++, v2.data);
            krn.setArg(pos++, v3.data);
            krn.setArg(pos++, v4.data);
            krn.setArg(pos++, v5.data);
            krn.setArg(pos++, v6.data);
            krn.setArg(pos++, m_alpha1);
            krn.setArg(pos++, m_alpha2);
            krn.setArg(pos++, m_alpha3);
            krn.setArg(pos++, m_alpha4);
            krn.setArg(pos++, m_alpha5);

            queue.enqueueNDRangeKernel(
                    krn, cl::NullRange, alignup(v1.n, wgsize), wgsize
                    );
        }

        typedef void result_type;
    };

};

static const value_type dt = 0.01;
static const value_type t_max = 100.0;

typedef clbuf<value_type> state_type;


struct sys_func
{
    const clbuf<value_type> &omega;

    sys_func( const clbuf<value_type> &omega ) : omega(omega) { }

    void operator()( const clbuf<value_type> &x , clbuf<value_type> &dxdt , value_type t )
    {
        static cl::Kernel krn(program, "oscillator_system");

        uint pos = 0;
        krn.setArg(pos++, x.n);
        krn.setArg(pos++, dxdt.data);
        krn.setArg(pos++, x.data);
        krn.setArg(pos++, omega.data);

        queue.enqueueNDRangeKernel(
                krn, cl::NullRange, alignup(x.n, wgsize), wgsize
                );
    }
};

int main(int argc, char *argv[]) {
    const size_t n = argc > 1 ? atoi(argv[1]) : 1024;
    const value_type epsilon = 6.0 / ( n * n ); // should be < 8/N^2 to see phase locking

    try {
        vex::Context vctx( vex::Filter::Exclusive( vex::Filter::Env && vex::Filter::Count(1) ) );
        if (!vctx) throw std::runtime_error("No compute devices");

        std::cout << vctx << std::endl;

        ctx     = vctx.context(0);
        device  = vctx.device(0);
        queue   = vctx.queue(0);
        program = vex::build_sources(ctx, clbuf_source);


        std::vector< value_type > omega( n );
        std::vector< value_type > x( n );
        for( size_t i=0 ; i<n ; ++i )
        {
            x[i] = 2.0 * M_PI * drand48();
            omega[i] = double( n - i ) * epsilon; // decreasing frequencies
        }

        state_type X( x );
        state_type Omega( omega );

        odeint::runge_kutta4<
            state_type , value_type , state_type , value_type ,
                       odeint::vector_space_algebra , clbuf_operations
                           > stepper;

        odeint::integrate_const( stepper , sys_func( Omega ) , X , value_type( 0.0 ) , t_max , dt );


        queue.enqueueReadBuffer(X.data, CL_TRUE, 0, sizeof(value_type), x.data());
        std::cout << x[0] << std::endl;
    } catch (const cl::Error &e) {
        std::cerr << "OpenCL error: " << e << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
