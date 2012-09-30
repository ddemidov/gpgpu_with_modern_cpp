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
    "double2 system_function(\n"
    "    double omega,\n"
    "    double amp,\n"
    "    double offset,\n"
    "    double omega_d,\n"
    "    double dt,\n"
    "    double t,\n"
    "    double2 s\n"
    "    )\n"
    "{\n"
    "    double eps = offset + amp * cos(omega_d * t);\n"
    "\n"
    "    double2 dsdt;\n"
    "\n"
    "    dsdt.x = dt * (eps * s.x + omega * s.y);\n"
    "    dsdt.y = dt * (eps * s.y - omega * s.x);\n"
    "\n"
    "    return dsdt;\n"
    "}\n"
    "\n"
    "kernel void dumped_oscillator(\n"
    "    ulong  n,\n"
    "    global double *X,\n"
    "    global double *Y,\n"
    "    double omega,\n"
    "    double amp,\n"
    "    double offset,\n"
    "    double omega_d,\n"
    "    double dt,\n"
    "    double t_max\n"
    "    )\n"
    "{\n"
    "    double t;\n"
    "    double2 s;\n"
    "    double2 dsdt;\n"
    "    double2 k1, k2, k3, k4;\n"
    "    for(size_t gid = get_global_id(0),\n"
    "               grid_size = get_global_size(0);\n"
    "        gid < n; gid += grid_size)\n"
    "    {\n"
    "        s.x = X[gid];\n"
    "        s.y = Y[gid];\n"
    "\n"
    "        for(t = 0; t < t_max; t += dt) {\n"
    "            k1 = system_function(omega, amp, offset, omega_d, dt,\n"
    "                    t, s);\n"
    "\n"
    "            k2 = system_function(omega, amp, offset, omega_d, dt,\n"
    "                    t + 0.5 * dt, s + 0.5 * k1);\n"
    "\n"
    "            k3 = system_function(omega, amp, offset, omega_d, dt,\n"
    "                    t + 0.5 * dt, s + 0.5 * k2);\n"
    "\n"
    "            k4 = system_function(omega, amp, offset, omega_d, dt,\n"
    "                    t + dt, s + k3);\n"
    "\n"
    "            s += (k1 + 2 * k2 + 2 * k3 + k4) / 6;\n"
    "        }\n"
    "\n"
    "        X[gid] = s.x;\n"
    "        Y[gid] = s.y;\n"
    "    }\n"
    "}\n";

typedef double value_type;
typedef vex::multivector<value_type,2> state_type;

int main( int argc , char **argv ) {
    const value_type dt = 0.01;
    const value_type t_max = 100.0;

    try {
	size_t n = argc > 1 ? atoi(argv[1]) : 1024;

	vex::Context ctx( vex::Filter::Exclusive( vex::Filter::DoublePrecision && vex::Filter::Env ) );
	std::cout << ctx << std::endl;

	std::vector<value_type> x( 2 * n );
	std::generate( x.begin() , x.end() , drand48 );

	state_type X(ctx.queue(), n);
	vex::copy( x.begin() , x.begin() + n, X(0).begin() );
	vex::copy( x.begin() + n, x.end() , X(1).begin() );

	for(uint d = 0; d < ctx.size(); d++) {
	    if (size_t psize = static_cast<cl_ulong>(X(0).part_size(d))) {
		cl::Program program = vex::build_sources(ctx.context(d), source);
		cl::Kernel kernel(program, "dumped_oscillator");
		cl::Device device = ctx.device(d);

		size_t wgsize = vex::kernel_workgroup_size(kernel, device);
		size_t g_size = device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU ?
		    vex::alignup(psize, wgsize) :
		    device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() * wgsize * 4;

		uint pos = 0;
		kernel.setArg(pos++, psize);
		kernel.setArg(pos++, X(0)(d));
		kernel.setArg(pos++, X(1)(d));
		kernel.setArg(pos++, 1.0);
		kernel.setArg(pos++, 0.2);
		kernel.setArg(pos++, 0.0);
		kernel.setArg(pos++, 1.2);
		kernel.setArg(pos++, dt);
		kernel.setArg(pos++, t_max);

		ctx.queue(d).enqueueNDRangeKernel(
			kernel, cl::NullRange, g_size, wgsize
			);
	    }
	}

	vex::copy( X(0).begin(), X(0).end(), x.begin() );

	std::cout << x[0] << std::endl;

    } catch(const cl::Error &e) {
	using namespace vex;
	std::cout << e << std::endl;
    }
}
