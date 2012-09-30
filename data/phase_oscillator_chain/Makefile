dat=$(wildcard *.dat)

all: perfcmp.eps scaling.eps tahiti_vs_tesla.eps

perfcmp.eps: process_results.m $(dat)
	matlab -nosplash -nodesktop -r 'process_results; exit'
	reset

scaling.eps: vexcl_scaling.m $(dat)
	matlab -nosplash -nodesktop -r 'vexcl_scaling; exit'
	reset

tahiti_vs_tesla.eps: tahiti_vs_tesla.m $(dat)
	matlab -nosplash -nodesktop -r 'tahiti_vs_tesla; exit'
	reset
