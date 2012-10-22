dat=$(wildcard *.dat)

all: perfmtx.eps scaling.eps

scaling.eps: vexcl_scaling.m $(dat)
	matlab -nosplash -nodesktop -r 'vexcl_scaling; exit'
	reset

perfmtx.eps: perfmtx.m $(dat)
	matlab -nosplash -nodesktop -r 'perfmtx; exit'
	reset

