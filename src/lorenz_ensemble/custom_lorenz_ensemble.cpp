#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <random>
#include <algorithm>

#include <vexcl/vexcl.hpp>
#include <vexcl/exclusive.hpp>

static const char source[] = 
    "#if defined(cl_khr_fp64)\n"
    "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
    "#elif defined(cl_amd_fp64)\n"
    "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
    "#endif\n"
    "\n"
    "double3 system_function(\n"
    "    double r,\n"
    "    double sigma,\n"
    "    double b,\n"
    "    double dt,\n"
    "    double3 s\n"
    "    )\n"
    "{\n"
    "    double3 dsdt;\n"
    "\n"
    "    dsdt.x = sigma * (s.y - s.x);\n"
    "    dsdt.y = r * s.x - s.y - s.x * s.z;\n"
    "    dsdt.z = s.x * s.y - b * s.z;\n"
    "\n"
    "    return dt * dsdt;\n"
    "}\n"
    "\n"
    "kernel void lorenz_ensemble(\n"
    "    ulong  n,\n"
    "    global double *X,\n"
    "    global double *Y,\n"
    "    global double *Z,\n"
    "    const global double *R,\n"
    "    double sigma,\n"
    "    double b,\n"
    "    double dt\n"
    "    )\n"
    "{\n"
    "    double r;\n"
    "    double3 s;\n"
    "    double3 dsdt;\n"
    "    double3 k1, k2, k3, k4;\n"
    "    for(size_t gid = get_global_id(0),\n"
    "               grid_size = get_global_size(0);\n"
    "        gid < n; gid += grid_size)\n"
    "    {\n"
    "        r   = R[gid];\n"
    "        s.x = X[gid];\n"
    "        s.y = Y[gid];\n"
    "        s.z = Z[gid];\n"
    "\n"
    "        k1 = system_function(r, sigma, b, dt, s);\n"
    "        k2 = system_function(r, sigma, b, dt, s + 0.5 * k1);\n"
    "        k3 = system_function(r, sigma, b, dt, s + 0.5 * k2);\n"
    "        k4 = system_function(r, sigma, b, dt, s + k3);\n"
    "\n"
    "        s += (k1 + 2 * k2 + 2 * k3 + k4) / 6;\n"
    "\n"
    "        X[gid] = s.x;\n"
    "        Y[gid] = s.y;\n"
    "        Z[gid] = s.z;\n"
    "    }\n"
    "}\n";

typedef double value_type;
typedef vex::vector< value_type >    vector_type;
typedef vex::multivector< value_type, 3 > state_type;

int main( int argc , char **argv ) {
    const value_type sigma = 10.0;
    const value_type b = 8.0 / 3.0;
    const value_type dt = 0.01;
    const value_type t_max = 100.0;

    try {
	size_t n = argc > 1 ? atoi(argv[1]) : 1024;

	vex::Context ctx( vex::Filter::Exclusive( vex::Filter::DoublePrecision && vex::Filter::Env ) );
	std::cout << ctx << std::endl;

	value_type Rmin = 0.1 , Rmax = 50.0 , dR = ( Rmax - Rmin ) / value_type( n - 1 );
	std::vector<value_type> r( n );
	for( size_t i=0 ; i<n ; ++i ) r[i] = Rmin + dR * value_type( i );
	vector_type R(ctx.queue(), r);

	state_type X(ctx.queue(), n);
	X = 10.0;

	std::vector<cl::Kernel> kernel(ctx.size());
	std::vector<size_t> wgsize(ctx.size());
	std::vector<size_t> g_size(ctx.size());

	for(uint d = 0; d < ctx.size(); d++) {
	    if (size_t psize = X(0).part_size(d)) {
		cl::Program program = vex::build_sources(ctx.context(d), source);
		kernel[d] = cl::Kernel(program, "lorenz_ensemble");

		cl::Device device = ctx.device(d);
		wgsize[d] = vex::kernel_workgroup_size(kernel[d], device);
		g_size[d] = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() * wgsize[d] * 4;
	    }
	}

	for(value_type t = 0; t < t_max; t += dt) {
	    for(uint d = 0; d < ctx.size(); d++) {
		if (size_t psize = X(0).part_size(d)) {
		    uint pos = 0;
		    kernel[d].setArg(pos++, psize);
		    kernel[d].setArg(pos++, X(0)(d));
		    kernel[d].setArg(pos++, X(1)(d));
		    kernel[d].setArg(pos++, X(2)(d));
		    kernel[d].setArg(pos++, R(d));
		    kernel[d].setArg(pos++, sigma);
		    kernel[d].setArg(pos++, b);
		    kernel[d].setArg(pos++, dt);

		    ctx.queue(d).enqueueNDRangeKernel(
			    kernel[d], cl::NullRange, g_size[d], wgsize[d]
			    );
		}
	    }
	}

	vex::copy( X(0).begin(), X(0).end(), r.begin() );
	std::cout << r[0] << std::endl;

    } catch(const cl::Error &e) {
	using namespace vex;
	std::cout << e << std::endl;
    }
}
