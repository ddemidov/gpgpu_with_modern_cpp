#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <random>
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

size_t bytes_touched = 0;

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
"kernel void ham_system(\n"
"    ulong n, uint w, ulong pitch,\n"
"    global const int *col,\n"
"    global const real *val,\n"
"    global const real *x,\n"
"    global real *dx,\n"
"    real beta\n"
"    )\n"
"{\n"
"    for(size_t i = get_global_id(0); i < n; i += get_global_size(0)) {\n"
"        real X = x[i];\n"
"        real sum = -beta * X * X * X;\n"
"        for(size_t j = 0; j < w; j++) {\n"
"            int c = col[i + j * pitch];\n"
"            if (c != -1) sum += val[i + j * pitch] * x[c];\n"
"        }\n"
"        dx[i] = sum;\n"
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

            bytes_touched +=
                v1.n * sizeof(T1) +
                v2.n * sizeof(T2) +
                v3.n * sizeof(T3);
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

            bytes_touched +=
                v1.n * sizeof(T1) +
                v2.n * sizeof(T2) +
                v3.n * sizeof(T2) +
                v4.n * sizeof(T3);
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

            bytes_touched +=
                v1.n * sizeof(T1) +
                v2.n * sizeof(T2) +
                v3.n * sizeof(T2) +
                v4.n * sizeof(T2) +
                v5.n * sizeof(T3);
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

            bytes_touched +=
                v1.n * sizeof(T1) +
                v2.n * sizeof(T2) +
                v3.n * sizeof(T2) +
                v4.n * sizeof(T2) +
                v5.n * sizeof(T2) +
                v6.n * sizeof(T3);
        }

        typedef void result_type;
    };

};

static const value_type K = 0.1;
static const value_type beta = 0.01;
static const value_type t_max = 100.0;
static const value_type dt = 0.01;

struct index_modulus {
    int N;

    index_modulus(int n) : N(n) {}

    inline int operator()(int idx) const {
	if( idx <  0 ) return idx + N;
	if( idx >= N ) return idx - N;
	return idx;
    }
};

struct sys_func
{
    clbuf<int>        col;
    clbuf<value_type> val;
    size_t n, pitch;
    uint   w;

    sys_func(int n1, int n2) : n(n1 * n2), pitch(alignup(n, 16)), w(5) {
        std::vector<value_type> disorder( n );
        std::generate(disorder.begin(), disorder.end(), drand48);

        std::vector<int>        C(w * pitch, -1);
        std::vector<value_type> V(w * pitch);

        index_modulus index(n);

        for( int i=0 ; i < n1 ; ++i ) {
            for( int j=0 ; j < n2 ; ++j ) {
                int idx = i * n2 + j;
                int is[5] = { idx , index( idx + 1 ) , index( idx - 1 ) , index( idx - n2 ) , index( idx + n2 ) };
                std::sort( is , is + 5 );
                for( int k=0 ; k < 5 ; ++k ) {
                    C[idx + pitch * k] = is[k];
                    V[idx + pitch * k] = (is[k] == idx ? -disorder[idx] - 4.0 * K : K);
                }
            }
        }

        clbuf<int>       (C).swap(col);
        clbuf<value_type>(V).swap(val);
    }

    void operator()( const clbuf<value_type> &q , clbuf<value_type> &dp )
    {
        static cl::Kernel krn(program, "ham_system");

        uint pos = 0;
        krn.setArg(pos++, n);
        krn.setArg(pos++, w);
        krn.setArg(pos++, pitch);
        krn.setArg(pos++, col.data);
        krn.setArg(pos++, val.data);
        krn.setArg(pos++, q.data);
        krn.setArg(pos++, dp.data);
        krn.setArg(pos++, beta);

        queue.enqueueNDRangeKernel(
                krn, cl::NullRange, alignup(n, wgsize), wgsize
                );

        bytes_touched += 
            (sizeof(int) + 2 * sizeof(value_type)) * n * w +
            sizeof(value_type) * 2 * n;
    }
};

typedef clbuf<value_type> state_type;

int main(int argc, char *argv[]) {
    const size_t n1 = argc > 1 ? atoi(argv[1]) : 64;
    const size_t n2 = n1;
    const size_t n = n1 * n2;

    try {
        vex::Context vctx( vex::Filter::Exclusive( vex::Filter::Env && vex::Filter::Count(1) ) );
        if (!vctx) throw std::runtime_error("No compute devices");

        std::cout << vctx << std::endl;

        ctx     = vctx.context(0);
        device  = vctx.device(0);
        queue   = vctx.queue(0);
        program = vex::build_sources(ctx, clbuf_source);

        std::vector<value_type> q(n, 0);
        std::vector<value_type> p(n, 0);
        q[n1/2*n2 + n2/2] = 1;

        std::pair<state_type, state_type> X;
        state_type(q).swap(X.first);
        state_type(p).swap(X.second);

        odeint::symplectic_rkn_sb3a_mclachlan<
            state_type , state_type , value_type , state_type , state_type , value_type ,
                       odeint::vector_space_algebra , clbuf_operations
                           > stepper;

        sys_func sys(n1, n2);
        odeint::integrate_const( stepper , std::ref( sys ) , X , value_type(0.0) , t_max , dt );

        queue.enqueueReadBuffer(X.first.data, CL_TRUE, 0, sizeof(value_type), q.data());
        std::cout << q[0] << std::endl;
        std::cout << "bytes io: " << bytes_touched << std::endl;
    } catch (const cl::Error &e) {
        std::cerr << "OpenCL error: " << e << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
